//! C API for AgentMemDB — used by Go and other C-compatible languages.

#![allow(static_mut_refs)]

use agent_mem_db::{AgentMemDB, AgentMemDBDisk, DiskOptions, Episode};
use libc::{c_char, c_float, c_int, c_longlong, size_t};
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

static mut LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);

fn set_last_error(msg: &str) {
    if let Ok(s) = CString::new(msg) {
        if let Ok(mut guard) = unsafe { LAST_ERROR.lock() } {
            *guard = Some(s);
        }
    }
}

/// Free a string returned by the C API.
#[no_mangle]
pub extern "C" fn agent_mem_db_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Get last error message. Caller must not free; valid until next API call.
#[no_mangle]
pub extern "C" fn agent_mem_db_last_error() -> *const c_char {
    if let Ok(guard) = unsafe { LAST_ERROR.lock() } {
        if let Some(ref s) = *guard {
            return s.as_ptr();
        }
    }
    ptr::null()
}

/// Create a new AgentMemDB for the given embedding dimension.
#[no_mangle]
pub extern "C" fn agent_mem_db_new(dim: size_t) -> *mut Mutex<AgentMemDB> {
    if dim == 0 {
        set_last_error("dim must be > 0");
        return ptr::null_mut();
    }
    let db = AgentMemDB::new(dim);
    Box::into_raw(Box::new(Mutex::new(db)))
}

/// Free an AgentMemDB handle.
#[no_mangle]
pub extern "C" fn agent_mem_db_free(h: *mut Mutex<AgentMemDB>) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) };
    }
}

/// Get embedding dimension. Returns 0 if handle is null.
#[no_mangle]
pub extern "C" fn agent_mem_db_dim(h: *mut Mutex<AgentMemDB>) -> size_t {
    if h.is_null() {
        return 0;
    }
    let db = unsafe { &*h };
    db.lock().unwrap().dim() as size_t
}

/// Store an episode. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_store(
    h: *mut Mutex<AgentMemDB>,
    task_id: *const c_char,
    embedding: *const c_float,
    dim: size_t,
    reward: c_float,
) -> c_int {
    if h.is_null() || task_id.is_null() || embedding.is_null() {
        set_last_error("null pointer");
        return -1;
    }
    let task_id = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid task_id utf-8");
                return -1;
            }
        }
    };
    let emb: Vec<f32> = unsafe { std::slice::from_raw_parts(embedding, dim).to_vec() };
    let ep = Episode::new(&task_id, emb, reward);
    let db = unsafe { &*h };
    match db.lock().unwrap().store_episode(ep) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Query for similar episodes. Returns JSON string (caller frees with agent_mem_db_free_string).
/// dim: embedding dimension (must match DB).
#[no_mangle]
pub extern "C" fn agent_mem_db_query(
    h: *mut Mutex<AgentMemDB>,
    embedding: *const c_float,
    dim: size_t,
    min_reward: c_float,
    top_k: size_t,
) -> *mut c_char {
    if h.is_null() || embedding.is_null() {
        set_last_error("null pointer");
        return ptr::null_mut();
    }
    let emb: Vec<f32> = unsafe { std::slice::from_raw_parts(embedding, dim).to_vec() };
    let db = unsafe { &*h };
    match db.lock().unwrap().query_similar(&emb, min_reward, top_k) {
        Ok(episodes) => {
            let json = serde_json::to_string(&episodes).unwrap_or_else(|_| "[]".into());
            match CString::new(json) {
                Ok(s) => s.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Save to file. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_save(h: *mut Mutex<AgentMemDB>, path: *const c_char) -> c_int {
    if h.is_null() || path.is_null() {
        set_last_error("null pointer");
        return -1;
    }
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid path utf-8");
                return -1;
            }
        }
    };
    let db = unsafe { &*h };
    match db.lock().unwrap().save_to_file(Path::new(&path_str)) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Load from file. Returns new handle or null on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_load(path: *const c_char) -> *mut Mutex<AgentMemDB> {
    if path.is_null() {
        set_last_error("null pointer");
        return ptr::null_mut();
    }
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid path utf-8");
                return ptr::null_mut();
            }
        }
    };
    match AgentMemDB::load_from_file(Path::new(&path_str)) {
        Ok(db) => Box::into_raw(Box::new(Mutex::new(db))),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Prune episodes with timestamp older than cutoff (Unix ms). Returns number removed.
#[no_mangle]
pub extern "C" fn agent_mem_db_prune_older_than(
    h: *mut Mutex<AgentMemDB>,
    timestamp_cutoff_ms: c_longlong,
) -> size_t {
    if h.is_null() {
        return 0;
    }
    let db = unsafe { &*h };
    db.lock()
        .unwrap()
        .prune_older_than(timestamp_cutoff_ms as i64) as size_t
}

/// Prune to keep only n most recent episodes. Returns number removed.
#[no_mangle]
pub extern "C" fn agent_mem_db_prune_keep_newest(h: *mut Mutex<AgentMemDB>, n: size_t) -> size_t {
    if h.is_null() {
        return 0;
    }
    let db = unsafe { &*h };
    db.lock().unwrap().prune_keep_newest(n) as size_t
}

/// Prune to keep only n highest-reward episodes. Returns number removed.
#[no_mangle]
pub extern "C" fn agent_mem_db_prune_keep_highest_reward(
    h: *mut Mutex<AgentMemDB>,
    n: size_t,
) -> size_t {
    if h.is_null() {
        return 0;
    }
    let db = unsafe { &*h };
    db.lock().unwrap().prune_keep_highest_reward(n) as size_t
}

// --- AgentMemDBDisk ---

/// Open disk-backed DB. Returns null on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_open(
    path: *const c_char,
    dim: size_t,
) -> *mut Mutex<AgentMemDBDisk> {
    if path.is_null() || dim == 0 {
        set_last_error("null path or dim must be > 0");
        return ptr::null_mut();
    }
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid path utf-8");
                return ptr::null_mut();
            }
        }
    };
    match AgentMemDBDisk::open(Path::new(&path_str), dim) {
        Ok(db) => Box::into_raw(Box::new(Mutex::new(db))),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Open disk-backed DB with exact index and checkpoint. Returns null on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_open_exact_with_checkpoint(
    path: *const c_char,
    dim: size_t,
) -> *mut Mutex<AgentMemDBDisk> {
    if path.is_null() || dim == 0 {
        set_last_error("null path or dim must be > 0");
        return ptr::null_mut();
    }
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid path utf-8");
                return ptr::null_mut();
            }
        }
    };
    match AgentMemDBDisk::open_with_options(
        Path::new(&path_str),
        DiskOptions::exact_with_checkpoint(dim),
    ) {
        Ok(db) => Box::into_raw(Box::new(Mutex::new(db))),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Free disk-backed DB handle.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_free(h: *mut Mutex<AgentMemDBDisk>) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) };
    }
}

/// Store episode. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_store(
    h: *mut Mutex<AgentMemDBDisk>,
    task_id: *const c_char,
    embedding: *const c_float,
    dim: size_t,
    reward: c_float,
) -> c_int {
    if h.is_null() || task_id.is_null() || embedding.is_null() {
        set_last_error("null pointer");
        return -1;
    }
    let task_id = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                set_last_error("invalid task_id utf-8");
                return -1;
            }
        }
    };
    let emb: Vec<f32> = unsafe { std::slice::from_raw_parts(embedding, dim).to_vec() };
    let ep = Episode::new(&task_id, emb, reward);
    let db = unsafe { &*h };
    match db.lock().unwrap().store_episode(ep) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Query. Returns JSON string (caller frees). Null on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_query(
    h: *mut Mutex<AgentMemDBDisk>,
    embedding: *const c_float,
    dim: size_t,
    min_reward: c_float,
    top_k: size_t,
) -> *mut c_char {
    if h.is_null() || embedding.is_null() {
        set_last_error("null pointer");
        return ptr::null_mut();
    }
    let emb: Vec<f32> = unsafe { std::slice::from_raw_parts(embedding, dim).to_vec() };
    let db = unsafe { &*h };
    match db.lock().unwrap().query_similar(&emb, min_reward, top_k) {
        Ok(episodes) => {
            let json = serde_json::to_string(&episodes).unwrap_or_else(|_| "[]".into());
            match CString::new(json) {
                Ok(s) => s.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Checkpoint. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_checkpoint(h: *mut Mutex<AgentMemDBDisk>) -> c_int {
    if h.is_null() {
        return -1;
    }
    let db = unsafe { &*h };
    match db.lock().unwrap().checkpoint() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Prune older than. Returns count removed, or -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_prune_older_than(
    h: *mut Mutex<AgentMemDBDisk>,
    timestamp_cutoff_ms: c_longlong,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let db = unsafe { &*h };
    match db
        .lock()
        .unwrap()
        .prune_older_than(timestamp_cutoff_ms as i64)
    {
        Ok(n) => n as c_int,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Prune keep newest. Returns count removed, or -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_prune_keep_newest(
    h: *mut Mutex<AgentMemDBDisk>,
    n: size_t,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let db = unsafe { &*h };
    match db.lock().unwrap().prune_keep_newest(n) {
        Ok(r) => r as c_int,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Prune keep highest reward. Returns count removed, or -1 on error.
#[no_mangle]
pub extern "C" fn agent_mem_db_disk_prune_keep_highest_reward(
    h: *mut Mutex<AgentMemDBDisk>,
    n: size_t,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let db = unsafe { &*h };
    match db.lock().unwrap().prune_keep_highest_reward(n) {
        Ok(r) => r as c_int,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}
