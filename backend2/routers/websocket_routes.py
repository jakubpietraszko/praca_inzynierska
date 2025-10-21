from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from services.container import ServiceContainer, get_container

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    container: ServiceContainer = Depends(get_container),  # noqa: B008
):
    websocket_service = container.websocket_service
    await websocket_service.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_service.disconnect(websocket)
