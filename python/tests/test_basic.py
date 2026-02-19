import agent_mem_db_py as agent_mem_db
import tempfile
import os


def test_async_api():
    """Test async wrappers (store, query, save, load)."""
    import asyncio

    async def run():
        mem = agent_mem_db.AgentMemDB(8)
        ep = agent_mem_db.Episode(
            task_id="t1",
            state_embedding=[0.1] * 8,
            reward=1.0,
            metadata={},
        )
        async_mem = agent_mem_db.AgentMemDBAsync(mem)
        await async_mem.store_episode_async(ep)
        results = await async_mem.query_similar_async(
            [0.1] * 8, min_reward=0.0, top_k=5
        )
        assert len(results) == 1
        assert results[0].task_id == "t1"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        await async_mem.save_to_file_async(path)
        mem2 = await agent_mem_db.AgentMemDBAsync.load_from_file_async(path)
        results2 = mem2.query_similar([0.1] * 8, min_reward=0.0, top_k=5)
        assert len(results2) == 1
        os.remove(path)

    asyncio.run(run())


def test_store_and_query():
    db = agent_mem_db.AgentMemDB(8)
    ep1 = agent_mem_db.Episode(
        task_id="t1",
        state_embedding=[0.1]*8,
        reward=1.0,
        metadata={}
    )
    ep2 = agent_mem_db.Episode(
        task_id="t2",
        state_embedding=[0.2]*8,
        reward=0.5,
        metadata={}
    )
    db.store_episode(ep1)
    db.store_episode(ep2)
    results = db.query_similar([0.1]*8, min_reward=0.0, top_k=2)
    assert len(results) == 2
    assert results[0].reward >= results[1].reward

def test_prune_older_than():
    """Test time-based pruning."""
    db = agent_mem_db.AgentMemDB(8)
    db.store_episode(
        agent_mem_db.Episode(
            task_id="old",
            state_embedding=[0.1] * 8,
            reward=1.0,
            timestamp=1000,
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="new",
            state_embedding=[0.1] * 8,
            reward=0.9,
            timestamp=3000,
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="no_ts",
            state_embedding=[0.1] * 8,
            reward=0.8,
            metadata={},
        )
    )
    removed = db.prune_older_than(2000)
    assert removed == 1
    results = db.query_similar([0.1] * 8, min_reward=0.0, top_k=5)
    assert len(results) == 2
    task_ids = [r.task_id for r in results]
    assert "new" in task_ids
    assert "no_ts" in task_ids
    assert "old" not in task_ids


def test_prune_keep_newest():
    """Test count-based pruning."""
    db = agent_mem_db.AgentMemDB(8)
    db.store_episode(
        agent_mem_db.Episode(
            task_id="a",
            state_embedding=[0.1] * 8,
            reward=0.9,
            timestamp=1000,
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="b",
            state_embedding=[0.1] * 8,
            reward=0.8,
            timestamp=2000,
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="c",
            state_embedding=[0.1] * 8,
            reward=0.7,
            timestamp=3000,
        )
    )
    removed = db.prune_keep_newest(2)
    assert removed == 1
    results = db.query_similar([0.1] * 8, min_reward=0.0, top_k=5)
    assert len(results) == 2
    task_ids = [r.task_id for r in results]
    assert "b" in task_ids
    assert "c" in task_ids
    assert "a" not in task_ids


def test_prune_keep_highest_reward():
    """Test importance-based pruning."""
    db = agent_mem_db.AgentMemDB(8)
    db.store_episode(
        agent_mem_db.Episode(
            task_id="low",
            state_embedding=[0.1] * 8,
            reward=0.3,
            metadata={},
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="high",
            state_embedding=[0.1] * 8,
            reward=0.9,
            metadata={},
        )
    )
    db.store_episode(
        agent_mem_db.Episode(
            task_id="mid",
            state_embedding=[0.1] * 8,
            reward=0.5,
            metadata={},
        )
    )
    removed = db.prune_keep_highest_reward(2)
    assert removed == 1
    results = db.query_similar([0.1] * 8, min_reward=0.0, top_k=5)
    assert len(results) == 2
    task_ids = [r.task_id for r in results]
    assert "high" in task_ids
    assert "mid" in task_ids
    assert "low" not in task_ids


def test_save_and_load_roundtrip():
    db = agent_mem_db.AgentMemDB(8)
    ep = agent_mem_db.Episode(
        task_id="t1",
        state_embedding=[0.1]*8,
        reward=1.0,
        metadata={}
    )
    db.store_episode(ep)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    db.save_to_file(path)
    db2 = agent_mem_db.AgentMemDB.load_from_file(path)
    results = db2.query_similar([0.1]*8, min_reward=0.0, top_k=1)
    assert len(results) == 1
    assert results[0].task_id == "t1"
    os.remove(path)


def test_disk_checkpoint_fast_restart():
    """Test ExactIndex checkpoint for fast restart."""
    tmpdir = os.path.join(tempfile.gettempdir(), "agent_mem_db_checkpoint_test")
    try:
        import shutil

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        db = agent_mem_db.AgentMemDBDisk.open_exact_with_checkpoint(tmpdir, 8)
        db.store_episode(
            agent_mem_db.Episode(
                task_id="t1",
                state_embedding=[0.1] * 8,
                reward=0.7,
                metadata={},
            )
        )
        db.store_episode(
            agent_mem_db.Episode(
                task_id="t2",
                state_embedding=[0.1] * 8,
                reward=0.8,
                metadata={},
            )
        )
        db.checkpoint()
        del db

        db2 = agent_mem_db.AgentMemDBDisk.open_exact_with_checkpoint(tmpdir, 8)
        results = db2.query_similar([0.1] * 8, min_reward=0.5, top_k=5)
        assert len(results) == 2
        assert os.path.exists(os.path.join(tmpdir, "exact_checkpoint.json"))
    finally:
        import shutil

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
