import asyncio
import json
import threading
import time
from typing import Dict, Optional, Set

from .config import (
    MAX_BROADCAST_FPS,
    MAX_IR_BROADCAST_FPS,
    WS_HOST,
    WS_PATH,
    WS_PORT,
    log,
)

try:
    from websockets.server import serve as ws_serve

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    ws_serve = None
    print("[WARN] 'websockets' not installed - run: pip install websockets")


class WSBroadcaster:
    def __init__(self):
        self._clients: Set = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_bcast_by_type: Dict[str, float] = {}
        self._min_interval_by_type = {
            "frame": 1.0 / MAX_BROADCAST_FPS,
            "ir": 1.0 / MAX_IR_BROADCAST_FPS,
        }

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def register(self, ws):
        with self._lock:
            self._clients.add(ws)
        log.info(f"[WS ] Client connected  (total={len(self._clients)})")

    def unregister(self, ws):
        with self._lock:
            self._clients.discard(ws)
        log.info(f"[WS ] Client disconnected (remaining={len(self._clients)})")

    def broadcast_json(self, payload: dict):
        if not self._loop:
            return

        payload_type = payload.get("type", "")
        now = time.monotonic()
        min_interval = self._min_interval_by_type.get(payload_type, 0.0)
        last_bcast = self._last_bcast_by_type.get(payload_type, 0.0)
        if now - last_bcast < min_interval:
            return
        self._last_bcast_by_type[payload_type] = now

        with self._lock:
            clients = set(self._clients)
        if not clients:
            return

        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._do_broadcast(msg, clients), self._loop)

    async def _do_broadcast(self, msg: str, clients: set):
        dead = set()
        for ws in clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        if dead:
            with self._lock:
                self._clients -= dead


# ─────────────────────────────────────────────


class WSServer:
    def __init__(self, broadcaster: WSBroadcaster):
        self.broadcaster = broadcaster

    async def handler(self, websocket, path=None):
        self.broadcaster.register(websocket)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self.broadcaster.unregister(websocket)

    async def run(self):
        log.info(f"[WS ] Listening on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
        async with ws_serve(self.handler, WS_HOST, WS_PORT):
            await asyncio.Future()


# ─────────────────────────────────────────────
