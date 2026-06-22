import asyncio

from .config import AUDIO_STREAM_URL, PHONE_STREAM_URL, UDP_PORT_CAM, UDP_PORT_IR, log
from .pipeline import Pipeline
from .websocket_server import HAS_WEBSOCKETS, WSBroadcaster, WSServer


class RoverApplication:
    def __init__(self) -> None:
        self._broadcaster = WSBroadcaster()
        self._pipeline = Pipeline(self._broadcaster)
        self._server = WSServer(self._broadcaster)

    def run(self) -> None:
        if not HAS_WEBSOCKETS:
            print("Install websockets:  pip install websockets")
            return

        self._log_banner()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._broadcaster.set_loop(loop)

        self._pipeline.start()

        try:
            loop.run_until_complete(self._server.run())
        except KeyboardInterrupt:
            log.info("Shutting down.")

    def _log_banner(self) -> None:
        log.info("=" * 60)
        log.info("  ROVER WebSocket Bridge  [PHONE STREAM UPDATE]")
        log.info(f"  UDP IR Source: port {UDP_PORT_IR}")
        if PHONE_STREAM_URL:
            log.info(f"  Video Source: Phone Cam Stream ({PHONE_STREAM_URL})")
        else:
            log.info(f"  Video Source: UDP Port {UDP_PORT_CAM}")
        log.info(f"  Audio Source: Phone Audio Stream ({AUDIO_STREAM_URL})")
        log.info("=" * 60)
