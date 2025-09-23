import json
import asyncio
import threading
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketService:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

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
            "filename": filename
        }

        disconnected = set()
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(json.dumps(notification))
            except Exception as e:
                disconnected.add(connection)

        for connection in disconnected:
            self.disconnect(connection)

    def send_heatmap_notification_sync(self, dataset: str, model: str, label: str, heatmap_type: str, filename: str):
        if not self.active_connections:
            return

        def run_async_notification():
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    self.send_heatmap_notification(
                        dataset, model, label, heatmap_type, filename
                    )
                )
            except Exception as e:
                pass

        thread = threading.Thread(target=run_async_notification, daemon=True)
        thread.start()
