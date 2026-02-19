//! Hosted Memory Cloud — HTTP server for Agent Memory DB.
//!
//! Phase 1: In-process AgentMemDB per tenant, API key auth.
//!
//! Usage:
//!   AGENT_MEM_API_KEY=secret cargo run --package agent_mem_db_server
//!   curl -H "Authorization: Bearer secret" -H "Content-Type: application/json" \
//!     -d '{"task_id":"t1","state_embedding":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"reward":0.9}' \
//!     http://localhost:8080/v1/episodes

use agent_mem_db::{AgentMemDB, AgentMemDBDisk, AgentMemError, DiskOptions, Episode, QueryOptions};
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Per-tenant backend: in-memory or disk-backed.
enum TenantBackend {
    InMemory(AgentMemDB),
    Disk(AgentMemDBDisk),
}

impl TenantBackend {
    fn store_episode(&mut self, ep: Episode) -> Result<(), AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => db.store_episode(ep),
            TenantBackend::Disk(db) => db.store_episode(ep),
        }
    }

    fn store_episodes(&mut self, episodes: Vec<Episode>) -> Result<(), AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => db.store_episodes(episodes),
            TenantBackend::Disk(db) => {
                for ep in episodes {
                    db.store_episode(ep)?;
                }
                Ok(())
            }
        }
    }

    fn query_similar_with_options(
        &self,
        embedding: &[f32],
        opts: QueryOptions,
    ) -> Result<Vec<Episode>, AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => db.query_similar_with_options(embedding, opts),
            TenantBackend::Disk(db) => db.query_similar_with_options(embedding, opts),
        }
    }

    fn prune_older_than(&mut self, ts: i64) -> Result<usize, AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => Ok(db.prune_older_than(ts)),
            TenantBackend::Disk(db) => db.prune_older_than(ts),
        }
    }

    fn prune_keep_newest(&mut self, n: usize) -> Result<usize, AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => Ok(db.prune_keep_newest(n)),
            TenantBackend::Disk(db) => db.prune_keep_newest(n),
        }
    }

    fn prune_keep_highest_reward(&mut self, n: usize) -> Result<usize, AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => Ok(db.prune_keep_highest_reward(n)),
            TenantBackend::Disk(db) => db.prune_keep_highest_reward(n),
        }
    }

    fn save_to_file(&self, path: &std::path::Path) -> Result<(), AgentMemError> {
        match self {
            TenantBackend::InMemory(db) => db.save_to_file(path),
            TenantBackend::Disk(_) => {
                // Disk backend is already persisted; save is a no-op
                Ok(())
            }
        }
    }

    fn checkpoint(&mut self) -> Result<(), AgentMemError> {
        match self {
            TenantBackend::InMemory(_) => Ok(()),
            TenantBackend::Disk(db) => db.checkpoint(),
        }
    }
}

/// Per-tenant DB. Key: tenant_id (from API key).
type TenantDB = Arc<RwLock<HashMap<String, TenantBackend>>>;

/// Simple in-memory metrics for observability (Prometheus-style).
#[derive(Clone)]
struct Metrics {
    requests_total: Arc<AtomicU64>,
    store_episodes_total: Arc<AtomicU64>,
    query_total: Arc<AtomicU64>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            requests_total: Arc::new(AtomicU64::new(0)),
            store_episodes_total: Arc::new(AtomicU64::new(0)),
            query_total: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Per-tenant rate limit: (request_count, window_start)
type RateLimitStore = Arc<RwLock<HashMap<String, (u64, Instant)>>>;

/// Audit log entry (JSONL).
#[derive(Serialize)]
struct AuditEntry {
    ts: String,
    tenant_id: String,
    op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    episode_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

fn audit_log(state: &AppState, tenant_id: &str, op: &str, task_id: Option<&str>, episode_count: Option<usize>, path: Option<&str>) {
    if let Some(ref audit) = state.audit_log {
        let entry = AuditEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            tenant_id: tenant_id.to_string(),
            op: op.to_string(),
            task_id: task_id.map(String::from),
            episode_count,
            path: path.map(String::from),
        };
        let audit = audit.clone();
        let line = serde_json::to_string(&entry).unwrap_or_else(|_| "{}".into());
        tokio::task::spawn_blocking(move || {
            use std::io::Write;
            if let Ok(mut guard) = audit.write() {
                if let Some(ref mut w) = *guard {
                    let _ = writeln!(w, "{}", line);
                    let _ = w.flush();
                }
            }
        });
    }
}

#[derive(Clone)]
struct AppState {
    tenants: TenantDB,
    default_dim: usize,
    data_dir: Option<PathBuf>,
    api_key: Option<String>,
    metrics: Metrics,
    rate_limit: Option<(RateLimitStore, u64, Duration)>,
    audit_log: Option<Arc<std::sync::RwLock<Option<std::fs::File>>>>,
}

#[derive(Deserialize)]
struct StoreEpisodeRequest {
    task_id: String,
    state_embedding: Vec<f32>,
    reward: f32,
    #[serde(default)]
    metadata: serde_json::Value,
    #[serde(default)]
    timestamp: Option<i64>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

#[derive(Serialize)]
struct StoreEpisodeResponse {
    id: String,
}

#[derive(Deserialize)]
struct StoreEpisodesRequest {
    episodes: Vec<StoreEpisodeRequest>,
}

#[derive(Serialize)]
struct StoreEpisodesResponse {
    ids: Vec<String>,
}

#[derive(Deserialize)]
struct QuerySimilarRequest {
    query_embedding: Vec<f32>,
    #[serde(default)]
    min_reward: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    tags_any: Option<Vec<String>>,
    #[serde(default)]
    tags_all: Option<Vec<String>>,
    #[serde(default)]
    task_id_prefix: Option<String>,
    #[serde(default)]
    time_after: Option<i64>,
    #[serde(default)]
    time_before: Option<i64>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

fn default_top_k() -> usize {
    5
}

#[derive(Serialize)]
struct QuerySimilarResponse {
    episodes: Vec<Episode>,
}

#[derive(Deserialize)]
struct SaveRequest {
    path: String,
}

#[derive(Serialize)]
struct SaveResponse {
    ok: bool,
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
}

#[derive(Serialize)]
struct LoadResponse {
    ok: bool,
}

#[derive(Deserialize)]
struct PruneOlderThanRequest {
    timestamp_cutoff_ms: i64,
}

#[derive(Serialize)]
struct PruneResponse {
    removed: usize,
}

#[derive(Deserialize)]
struct PruneKeepNewestRequest {
    n: usize,
}

#[derive(Deserialize)]
struct PruneKeepHighestRewardRequest {
    n: usize,
}

/// Resolve tenant from API key. For Phase 1, API key maps 1:1 to tenant_id.
fn tenant_from_key(api_key: &str) -> String {
    api_key.to_string()
}

/// Sanitize tenant_id for use as a path component.
fn sanitize_tenant_path(tenant_id: &str) -> String {
    tenant_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Create a new tenant backend. When data_dir is set, uses AgentMemDBDisk with checkpoint.
fn create_tenant_backend(
    data_dir: Option<&PathBuf>,
    tenant_id: &str,
    dim: usize,
) -> Result<TenantBackend, AgentMemError> {
    if let Some(dir) = data_dir {
        let safe = sanitize_tenant_path(tenant_id);
        let tenant_path = dir.join(safe);
        let db = AgentMemDBDisk::open_with_options(
            tenant_path,
            DiskOptions::exact_with_checkpoint(dim),
        )?;
        Ok(TenantBackend::Disk(db))
    } else {
        Ok(TenantBackend::InMemory(AgentMemDB::new(dim)))
    }
}

/// Extract API key from Authorization header or X-API-Key.
fn extract_api_key(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(auth) = headers.get("Authorization") {
        if let Ok(s) = auth.to_str() {
            if let Some(key) = s.strip_prefix("Bearer ") {
                return Some(key.to_string());
            }
        }
    }
    if let Some(key) = headers.get("X-API-Key") {
        if let Ok(s) = key.to_str() {
            return Some(s.to_string());
        }
    }
    None
}

/// Auth middleware: validate API key and insert tenant_id into extensions.
async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, Response> {
    let key = extract_api_key(request.headers()).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "Missing Authorization: Bearer <key> or X-API-Key"})),
        )
            .into_response()
    })?;

    if let Some(ref expected) = state.api_key {
        if key != *expected {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid API key"})),
            )
                .into_response());
        }
    }

    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    let tenant_id = tenant_from_key(&key);
    let mut request = request;
    request.extensions_mut().insert(tenant_id);
    Ok(next.run(request).await)
}

/// Rate limit middleware: per-tenant fixed window. Runs after auth (requires tenant_id in extensions).
async fn rate_limit_middleware(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, Response> {
    let Some(ref config) = state.rate_limit else {
        return Ok(next.run(request).await);
    };
    let (store, max_per_window, window) = config;
    let tenant_id = request
        .extensions()
        .get::<String>()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());

    let now = Instant::now();
    let mut guard = store.write().await;
    let (count, window_start) = guard.entry(tenant_id.clone()).or_insert((0, now));
    if now.duration_since(*window_start) >= *window {
        *count = 0;
        *window_start = now;
    }
    *count += 1;
    let current = *count;
    drop(guard);

    if current > *max_per_window {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({"error": "Rate limit exceeded"})),
        )
            .into_response());
    }
    Ok(next.run(request).await)
}

async fn health() -> &'static str {
    "ok"
}

async fn dashboard(State(state): State<AppState>) -> Html<String> {
    let requests = state.metrics.requests_total.load(Ordering::Relaxed);
    let store_episodes = state.metrics.store_episodes_total.load(Ordering::Relaxed);
    let queries = state.metrics.query_total.load(Ordering::Relaxed);
    let tenants = state.tenants.read().await.len();

    let rate_limit_str = state
        .rate_limit
        .as_ref()
        .map(|(_, max, dur)| format!("{} req / {}s", max, dur.as_secs()))
        .unwrap_or_else(|| "disabled".to_string());
    let audit_str = if state.audit_log.is_some() {
        "enabled"
    } else {
        "disabled"
    };
    let api_key_str = if state.api_key.is_some() {
        "set"
    } else {
        "not set (dev)"
    };

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Memory DB — Dashboard</title>
  <style>
    :root {{ font-family: system-ui, -apple-system, sans-serif; font-size: 16px; }}
    body {{ max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 1.5rem; }}
    section {{ margin-bottom: 1.5rem; }}
    h2 {{ font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #666; margin-bottom: 0.5rem; }}
    .metric {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #eee; }}
    .metric span:last-child {{ font-variant-numeric: tabular-nums; font-weight: 500; }}
    .status {{ color: #0a0; font-weight: 500; }}
    a {{ color: #0066cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Agent Memory DB</h1>

  <section>
    <h2>Health</h2>
    <div class="metric"><span>Status</span><span class="status">ok</span></div>
    <div class="metric"><span><a href="/health">/health</a></span><span></span></div>
    <div class="metric"><span><a href="/metrics">/metrics</a></span><span>Prometheus</span></div>
  </section>

  <section>
    <h2>Usage</h2>
    <div class="metric"><span>API requests</span><span>{}</span></div>
    <div class="metric"><span>Episodes stored</span><span>{}</span></div>
    <div class="metric"><span>Queries</span><span>{}</span></div>
    <div class="metric"><span>Active tenants</span><span>{}</span></div>
  </section>

  <section>
    <h2>Config</h2>
    <div class="metric"><span>Embedding dim</span><span>{}</span></div>
    <div class="metric"><span>API key</span><span>{}</span></div>
    <div class="metric"><span>Rate limit</span><span>{}</span></div>
    <div class="metric"><span>Audit log</span><span>{}</span></div>
    <div class="metric"><span>Data dir</span><span>{}</span></div>
  </section>
</body>
</html>"##,
        requests,
        store_episodes,
        queries,
        tenants,
        state.default_dim,
        api_key_str,
        rate_limit_str,
        audit_str,
        state
            .data_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "—".to_string())
    );
    Html(html)
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let requests = state.metrics.requests_total.load(Ordering::Relaxed);
    let store_episodes = state.metrics.store_episodes_total.load(Ordering::Relaxed);
    let queries = state.metrics.query_total.load(Ordering::Relaxed);
    let tenants = state.tenants.read().await.len();
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        format!(
            "# HELP agent_mem_requests_total Total authenticated API requests\n\
             # TYPE agent_mem_requests_total counter\n\
             agent_mem_requests_total {}\n\
             # HELP agent_mem_store_episodes_total Total episodes stored\n\
             # TYPE agent_mem_store_episodes_total counter\n\
             agent_mem_store_episodes_total {}\n\
             # HELP agent_mem_query_total Total similarity queries\n\
             # TYPE agent_mem_query_total counter\n\
             agent_mem_query_total {}\n\
             # HELP agent_mem_tenants_active Active tenant count\n\
             # TYPE agent_mem_tenants_active gauge\n\
             agent_mem_tenants_active {}\n",
            requests, store_episodes, queries, tenants
        ),
    )
}

async fn store_episode(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<StoreEpisodeRequest>,
) -> Result<Json<StoreEpisodeResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut ep = Episode::new(&req.task_id, req.state_embedding.clone(), req.reward);
    ep.metadata = req.metadata;
    ep.timestamp = req.timestamp;
    ep.tags = req.tags;
    ep.source = req.source;
    ep.user_id = req.user_id;
    let id = ep.id.to_string();

    let mut tenants = state.tenants.write().await;
    let db = match tenants.entry(tenant_id.clone()) {
        std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => {
            let backend = create_tenant_backend(state.data_dir.as_ref(), &tenant_id, state.default_dim)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))?;
            v.insert(backend)
        }
    };

    db.store_episode(ep).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
    })?;

    state.metrics.store_episodes_total.fetch_add(1, Ordering::Relaxed);
    audit_log(&state, &tenant_id, "store_episode", Some(&req.task_id), Some(1), None);
    Ok(Json(StoreEpisodeResponse { id }))
}

async fn store_episodes(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<StoreEpisodesRequest>,
) -> Result<Json<StoreEpisodesResponse>, (StatusCode, Json<serde_json::Value>)> {
    let episodes: Vec<Episode> = req
        .episodes
        .into_iter()
        .map(|e| {
            let mut ep = Episode::new(&e.task_id, e.state_embedding, e.reward);
            ep.metadata = e.metadata;
            ep.timestamp = e.timestamp;
            ep.tags = e.tags;
            ep.source = e.source;
            ep.user_id = e.user_id;
            ep
        })
        .collect();
    let ids: Vec<String> = episodes.iter().map(|e| e.id.to_string()).collect();

    let mut tenants = state.tenants.write().await;
    let db = match tenants.entry(tenant_id.clone()) {
        std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => {
            let backend = create_tenant_backend(state.data_dir.as_ref(), &tenant_id, state.default_dim)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))?;
            v.insert(backend)
        }
    };

    db.store_episodes(episodes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
    })?;

    state.metrics.store_episodes_total.fetch_add(ids.len() as u64, Ordering::Relaxed);
    audit_log(&state, &tenant_id, "store_episodes", None, Some(ids.len()), None);
    Ok(Json(StoreEpisodesResponse { ids }))
}

async fn query_similar(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<QuerySimilarRequest>,
) -> Result<Json<QuerySimilarResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut tenants = state.tenants.write().await;
    let db = if let Some(backend) = tenants.get_mut(&tenant_id) {
        backend
    } else if let Some(ref data_dir) = state.data_dir {
        let meta_path = data_dir.join(sanitize_tenant_path(&tenant_id)).join("meta.json");
        if !meta_path.exists() {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
            ));
        }
        let backend = create_tenant_backend(Some(data_dir), &tenant_id, state.default_dim)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))))?;
        tenants.insert(tenant_id.clone(), backend);
        tenants.get_mut(&tenant_id).unwrap()
    } else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
        ));
    };

    let mut opts = QueryOptions::new(req.min_reward, req.top_k);
    if let Some(tags) = req.tags_any {
        if !tags.is_empty() {
            opts = opts.tags_any(tags);
        }
    }
    if let Some(tags) = req.tags_all {
        if !tags.is_empty() {
            opts = opts.tags_all(tags);
        }
    }
    if let Some(ref prefix) = req.task_id_prefix {
        opts = opts.task_id_prefix(prefix.clone());
    }
    if let Some(ts) = req.time_after {
        opts = opts.time_after(ts);
    }
    if let Some(ts) = req.time_before {
        opts = opts.time_before(ts);
    }
    if let Some(ref s) = req.source {
        opts = opts.source(s.clone());
    }
    if let Some(ref u) = req.user_id {
        opts = opts.user_id(u.clone());
    }

    let episodes = db
        .query_similar_with_options(&req.query_embedding, opts)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        })?;

    state.metrics.query_total.fetch_add(1, Ordering::Relaxed);
    audit_log(&state, &tenant_id, "query", None, None, None);
    Ok(Json(QuerySimilarResponse { episodes }))
}

async fn save(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<SaveRequest>,
) -> Result<Json<SaveResponse>, (StatusCode, Json<serde_json::Value>)> {
    let tenants = state.tenants.read().await;
    let db = tenants.get(&tenant_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
    ))?;

    let path = state
        .data_dir
        .as_ref()
        .map(|d| d.join(&req.path))
        .unwrap_or_else(|| PathBuf::from(&req.path));

    db.save_to_file(&path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Save failed: {}", e)})),
        )
    })?;

    audit_log(&state, &tenant_id, "save", None, None, Some(req.path.as_str()));
    Ok(Json(SaveResponse { ok: true }))
}

async fn load(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<LoadResponse>, (StatusCode, Json<serde_json::Value>)> {
    if state.data_dir.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Load not supported when using disk-backed storage (AGENT_MEM_DATA_DIR)"})),
        ));
    }

    let path = PathBuf::from(&req.path);
    let db = AgentMemDB::load_from_file(&path).map_err(|e| (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": format!("Load failed: {}", e)})),
    ))?;

    let mut tenants = state.tenants.write().await;
    tenants.insert(tenant_id.clone(), TenantBackend::InMemory(db));

    audit_log(&state, &tenant_id, "load", None, None, Some(req.path.as_str()));
    Ok(Json(LoadResponse { ok: true }))
}

async fn prune_older_than(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<PruneOlderThanRequest>,
) -> Result<Json<PruneResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut tenants = state.tenants.write().await;
    let db = tenants.get_mut(&tenant_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
    ))?;

    let removed = db.prune_older_than(req.timestamp_cutoff_ms).map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    ))?;
    audit_log(&state, &tenant_id, "prune_older_than", None, Some(removed), None);
    Ok(Json(PruneResponse { removed }))
}

async fn prune_keep_newest(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<PruneKeepNewestRequest>,
) -> Result<Json<PruneResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut tenants = state.tenants.write().await;
    let db = tenants.get_mut(&tenant_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
    ))?;

    let removed = db.prune_keep_newest(req.n).map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    ))?;
    audit_log(&state, &tenant_id, "prune_keep_newest", None, Some(removed), None);
    Ok(Json(PruneResponse { removed }))
}

async fn prune_keep_highest_reward(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
    Json(req): Json<PruneKeepHighestRewardRequest>,
) -> Result<Json<PruneResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut tenants = state.tenants.write().await;
    let db = tenants.get_mut(&tenant_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
    ))?;

    let removed = db.prune_keep_highest_reward(req.n).map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    ))?;
    audit_log(&state, &tenant_id, "prune_keep_highest_reward", None, Some(removed), None);
    Ok(Json(PruneResponse { removed }))
}

#[derive(Serialize)]
struct CheckpointResponse {
    ok: bool,
}

async fn checkpoint(
    State(state): State<AppState>,
    axum::extract::Extension(tenant_id): axum::extract::Extension<String>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<serde_json::Value>)> {
    let mut tenants = state.tenants.write().await;
    let db = tenants.get_mut(&tenant_id).ok_or((
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "No episodes stored for this tenant yet"})),
    ))?;

    db.checkpoint().map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    ))?;

    audit_log(&state, &tenant_id, "checkpoint", None, None, None);
    Ok(Json(CheckpointResponse { ok: true }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let api_key = std::env::var("AGENT_MEM_API_KEY").ok();
    let default_dim: usize = std::env::var("AGENT_MEM_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let data_dir = std::env::var("AGENT_MEM_DATA_DIR").ok().map(PathBuf::from);

    let rate_limit = std::env::var("AGENT_MEM_RATE_LIMIT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(|max_per_window| {
            let window_secs = std::env::var("AGENT_MEM_RATE_WINDOW_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            (
                Arc::new(RwLock::new(HashMap::new())),
                max_per_window,
                Duration::from_secs(window_secs),
            )
        });

    let audit_log = std::env::var("AGENT_MEM_AUDIT_LOG")
        .ok()
        .and_then(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()
                .map(|f| Arc::new(std::sync::RwLock::new(Some(f))))
        });

    let state = AppState {
        tenants: Arc::new(RwLock::new(HashMap::new())),
        default_dim,
        data_dir,
        api_key: api_key.clone(),
        metrics: Metrics::default(),
        rate_limit,
        audit_log,
    };

    let cors = CorsLayer::permissive();
    let trace = TraceLayer::new_for_http()
        .on_request(|req: &Request<_>, _: &tracing::Span| {
            tracing::info!(method = %req.method(), uri = %req.uri(), "request");
        })
        .on_response(|res: &Response, latency: std::time::Duration, _: &tracing::Span| {
            tracing::info!(status = %res.status(), latency_ms = %latency.as_millis(), "response");
        });

    let rate_limit_enabled = state.rate_limit.is_some();
    let audit_enabled = state.audit_log.is_some();

    let v1_routes = Router::new()
        .route("/episodes", post(store_episode))
        .route("/episodes/batch", post(store_episodes))
        .route("/query", post(query_similar))
        .route("/save", post(save))
        .route("/load", post(load))
        .route("/prune/older-than", post(prune_older_than))
        .route("/prune/keep-newest", post(prune_keep_newest))
        .route("/prune/keep-highest-reward", post(prune_keep_highest_reward))
        .route("/checkpoint", post(checkpoint))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state.clone());

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/dashboard", get(dashboard))
        .nest("/v1", v1_routes)
        .layer(trace)
        .layer(cors)
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Listening on http://{}", addr);
    if api_key.is_none() {
        tracing::warn!("AGENT_MEM_API_KEY not set — all API keys accepted (dev only)");
    }
    if rate_limit_enabled {
        tracing::info!("Rate limiting enabled (AGENT_MEM_RATE_LIMIT)");
    }
    if audit_enabled {
        tracing::info!("Audit logging enabled (AGENT_MEM_AUDIT_LOG)");
    }

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
