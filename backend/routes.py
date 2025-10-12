"""API routes for the Open WebUI Chat Analyzer backend."""

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from .models import Chat, DatasetMeta, Message, OpenWebUISyncRequest, UploadResponse, User
from .services import DataService, get_data_service

router = APIRouter(prefix="/api/v1", tags=["data"])


@router.get("/chats", response_model=List[Chat])
def list_chats(service: DataService = Depends(get_data_service)) -> List[Chat]:
    """Return all chat metadata."""
    chats = service.get_chats()
    return [Chat(**chat) for chat in chats]


@router.get("/messages", response_model=List[Message])
def list_messages(service: DataService = Depends(get_data_service)) -> List[Message]:
    """Return all chat messages."""
    messages = service.get_messages()
    return [Message(**message) for message in messages]


@router.get("/users", response_model=List[User])
def list_users(service: DataService = Depends(get_data_service)) -> List[User]:
    """Return optional user metadata."""
    users = service.get_users()
    return [User(**user) for user in users]


@router.get("/datasets/meta", response_model=DatasetMeta)
def dataset_meta(service: DataService = Depends(get_data_service)) -> DatasetMeta:
    """Return metadata about the current dataset."""
    return service.get_meta()


@router.post(
    "/uploads/chat-export",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_chat_export(
    file: UploadFile = File(...),
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Upload a new chat export JSON file."""
    if file.content_type not in (
        "application/json",
        "text/json",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="File must be a JSON export.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    original_filename = file.filename or None

    try:
        service.update_chat_export(
            raw_bytes,
            source_label=None,
            persist_filename=original_filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(detail="Chat export uploaded successfully.", dataset=service.get_meta())


@router.post(
    "/uploads/users",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_users_csv(
    file: UploadFile = File(...),
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Upload a users CSV file."""
    if file.content_type not in (
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    original_filename = file.filename or None

    try:
        service.update_users(
            raw_bytes,
            source_label=None,
            persist_filename=original_filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(detail="Users CSV uploaded successfully.", dataset=service.get_meta())


@router.post(
    "/openwebui/sync",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
def sync_openwebui(
    payload: OpenWebUISyncRequest,
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Fetch chats and users directly from an Open WebUI instance and store them locally."""
    try:
        dataset = service.sync_from_openwebui(payload.hostname, payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return UploadResponse(detail="Open WebUI data synced successfully.", dataset=dataset)


@router.get("/summaries/status")
def summary_status(service: DataService = Depends(get_data_service)) -> dict:
    """Return the current status of the summary job."""
    return service.get_summary_status()


@router.post("/summaries/rebuild")
def rebuild_summaries(service: DataService = Depends(get_data_service)) -> dict:
    """Trigger an asynchronous rebuild of chat summaries."""
    status_state = service.rebuild_summaries()
    return {"ok": True, "status": status_state}
