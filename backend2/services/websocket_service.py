import contextlib
import json

from fastapi import WebSocket


class WebSocketService:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_heatmap_notification(self, dataset: str, model: str, label: str, heatmap_type: str, filename: str):
        if not self.active_connections:
            return

        notification = {
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
    ):
        if not self.active_connections:
            return

        with contextlib.suppress(Exception):
            await self.send_heatmap_notification(dataset, model, label, heatmap_type, filename)

        return
