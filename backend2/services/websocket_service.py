import builtins
import contextlib
import json

from fastapi import WebSocket

from interfaces.heatmap import HeatmapRequest


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

        notification: builtins.dict[builtins.str, builtins.str] = {
            "dataset": dataset,
            "model": model,
            "label": label,
            "heatmap_type": heatmap_type,
            "filename": filename,
        }

        disconnected = set()
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(json.dumps(notification))
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
        with contextlib.suppress(builtins.BaseException):
            await self.send_heatmap_notification_sync(
                dataset=request.dataset,
                model=request.model,
                label=label,
                filename=filename,
                heatmap_type=heatmap_type,
            )
