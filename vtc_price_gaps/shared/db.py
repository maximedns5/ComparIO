from __future__ import annotations
import asyncpg
import redis.asyncio as aioredis
from shared.config import cfg

_pg_pool = None
_redis_client = None

async def get_pg_pool() -> asyncpg.Pool:
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(cfg.database_url, min_size=2, max_size=10, command_timeout=60)
    return _pg_pool

async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(cfg.redis_url, encoding="utf-8", decode_responses=True)
    return _redis_client

async def close_connections():
    global _pg_pool, _redis_client
    if _pg_pool: await _pg_pool.close()
    if _redis_client: await _redis_client.aclose()