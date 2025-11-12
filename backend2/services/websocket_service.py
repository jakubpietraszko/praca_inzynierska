import asyncio
import builtins
import contextlib

from fastapi import WebSocket

from interfaces.heatmap import HeatmapRequest
from interfaces.notification import HeatmapNotification


class WebSocketService:
    def __init__(self) -> None:
        from services.container import Constants, get_constants

        self.active_connections: set[WebSocket] = set()
        self.constants: Constants = get_constants()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def send_heatmap_notification(self, dataset: str, model: str, label: str, heatmap_type: str, filename: str):
        if not self.active_connections:
            return

        notification = HeatmapNotification(
            dataset=dataset,
            model=model,
            label=label,
            heatmap_type=heatmap_type,
            filename=filename,
        )

        disconnected = set()
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(notification.model_dump_json())
            except Exception:
                disconnected.add(connection)

        for connection in disconnected:
            self.disconnect(connection)

    async def send_heatmap_notification_sync(
        self, dataset: str, model: str, label: str, heatmap_type: str, filename: str
    ) -> None:
        if not self.active_connections:
            return

        with contextlib.suppress(Exception):
            await self.send_heatmap_notification(dataset, model, label, heatmap_type, filename)

        return

    async def send_notification_supress_exceptions(
        self, request: HeatmapRequest, label: str, filename: str, heatmap_type: str
    ) -> None:
        asyncio.create_task(
            self._send_notification_background(
                request.dataset,
                request.model,
                label,
                filename,
                heatmap_type,
            )
        )
        await asyncio.sleep(0)

    async def _send_notification_background(
        self, dataset: str, model: str, label: str, filename: str, heatmap_type: str
    ) -> None:
        with contextlib.suppress(builtins.BaseException):
            await self._send_notification_immediate(dataset, model, label, filename, heatmap_type)

    async def _send_notification_immediate(
        self, dataset: str, model: str, label: str, filename: str, heatmap_type: str
    ) -> None:
        if not self.active_connections:
            return

        notification = HeatmapNotification(
            dataset=dataset,
            model=model,
            label=label,
            heatmap_type=heatmap_type,
            filename=filename,
        )

        disconnected = set()
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(notification.model_dump_json())
            except Exception:
                disconnected.add(connection)

        for connection in disconnected:
            self.disconnect(connection)
