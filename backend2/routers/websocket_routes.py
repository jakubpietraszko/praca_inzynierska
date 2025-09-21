from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from services.container import get_container, ServiceContainer

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    container: ServiceContainer = Depends(get_container)
):
    websocket_service = container.websocket_service
    await websocket_service.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_service.disconnect(websocket)
