from __future__ import annotations

import json
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.ws_protocol import event_to_dict

router = APIRouter(prefix="/api")


def _get_manager():
    from api.main import session_manager

    return session_manager


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = _get_manager()

    session = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "create_session":
                sid, session = await manager.create_session()
                await websocket.send_json({"type": "session_created", "session_id": sid})

            elif msg_type == "resume_session":
                result = await manager.resume_session(data["session_id"])
                if result:
                    sid, session = result
                    for msg in session.history:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role in ("user", "assistant") and content:
                            await websocket.send_json(
                                {"type": "history_message", "role": role, "content": content}
                            )
                    await websocket.send_json({"type": "session_resumed", "session_id": sid})
                else:
                    await websocket.send_json({"type": "error", "message": "Session not found"})

            elif msg_type == "message":
                if session is None:
                    sid, session = await manager.create_session()
                    await websocket.send_json({"type": "session_created", "session_id": sid})

                content = data.get("content", "")
                files = data.get("files")

                if files:
                    user_input = _build_multimodal_input(content, files)
                else:
                    user_input = content

                async for event in session.turn_stream(user_input):
                    msg = event_to_dict(event)
                    if msg is not None:
                        await websocket.send_json(msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def _build_multimodal_input(text: str, file_refs: list[dict]) -> list[dict]:
    """Build Anthropic-style multimodal content blocks from text + uploaded files."""
    import base64
    from pathlib import Path

    blocks: list[dict] = []

    for ref in file_refs:
        path = Path(ref.get("path", ""))
        if not path.is_file():
            continue

        if ref.get("type") == "image":
            data = base64.standard_b64encode(path.read_bytes()).decode()
            media_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_map.get(path.suffix.lower(), "image/png")
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        else:
            try:
                file_content = path.read_text(errors="replace")
                text += f"\n\n--- Contents of {ref.get('name', path.name)} ---\n{file_content}\n---"
            except Exception:
                pass

    blocks.append({"type": "text", "text": text})
    return blocks
