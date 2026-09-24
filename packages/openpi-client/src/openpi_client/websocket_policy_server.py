from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Optional

import websockets
import websockets.asyncio.server as _server
import websockets.frames

from openpi_client import base_policy, msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve an ``openpi_client`` policy over the standard msgpack WebSocket protocol."""

    def __init__(
        self,
        policy: base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: Optional[int] = None,  # noqa: UP045
        metadata: Optional[dict] = None,  # noqa: UP045
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=60,
            ping_timeout=120,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        previous_total = None
        while True:
            try:
                started = time.monotonic()
                observation = msgpack_numpy.unpackb(await websocket.recv())
                infer_started = time.monotonic()
                response = self._policy.infer(observation)
                infer_seconds = time.monotonic() - infer_started
                response["server_timing"] = {"infer_ms": infer_seconds * 1000}
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
