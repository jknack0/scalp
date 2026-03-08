"""Health monitoring endpoint using FastAPI."""

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.events import EventBus, EventType
from src.core.logging import get_logger

logger = get_logger("health")


class HealthMonitor:
    """Exposes GET /health with bot status.

    Runs uvicorn in a background asyncio task.
    """

    def __init__(
        self,
        event_bus: EventBus,
        risk_manager: "Any",  # avoid circular import
        port: int = 8080,
    ) -> None:
        self._bus = event_bus
        self._risk = risk_manager
        self._port = port
        self._start_time = time.monotonic()
        self._last_tick_time: float = 0.0
        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="MES Bot Health", docs_url=None, redoc_url=None)

        @app.get("/health")
        async def health() -> JSONResponse:
            now = time.monotonic()
            last_tick_age_ms = (
                (now - self._last_tick_time) * 1000
                if self._last_tick_time > 0
                else -1
            )
            return JSONResponse(
                {
                    "status": "halted" if self._risk.is_halted else "running",
                    "position": self._risk.current_position,
                    "daily_pnl_usd": round(self._risk.daily_pnl, 2),
                    "last_tick_age_ms": round(last_tick_age_ms, 1),
                    "is_halted": self._risk.is_halted,
                    "event_counts": {
                        k.name: v for k, v in self._bus.event_counts.items()
                    },
                    "uptime_seconds": round(now - self._start_time, 1),
                }
            )

        return app

    async def _on_tick(self, event: "Any") -> None:
        """EventBus callback to track last tick time."""
        self._last_tick_time = time.monotonic()

    async def start(self) -> None:
        """Start the health server as a background task."""
        import uvicorn

        self._bus.subscribe(EventType.TICK, self._on_tick)

        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.info("health_server_starting", port=self._port)
        await server.serve()
