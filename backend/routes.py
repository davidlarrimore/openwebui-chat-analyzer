"""API routes for the Open WebUI Chat Analyzer backend."""

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from .models import Chat, DatasetMeta, Message, ModelInfo, OpenWebUISyncRequest, UploadResponse, User
from .services import DataService, get_data_service

router = APIRouter(prefix="/api/v1", tags=["data"])


@router.get("/chats", response_model=List[Chat])
def list_chats(service: DataService = Depends(get_data_service)) -> List[Chat]:
    """Return all chat metadata.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[Chat]: Chats converted into the API schema.
    """
    chats = service.get_chats()
    return [Chat(**chat) for chat in chats]


@router.get("/messages", response_model=List[Message])
def list_messages(service: DataService = Depends(get_data_service)) -> List[Message]:
    """Return the flattened list of chat messages.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[Message]: Serialized messages compliant with the API schema.
    """
    messages = service.get_messages()
    return [Message(**message) for message in messages]


@router.get("/users", response_model=List[User])
def list_users(service: DataService = Depends(get_data_service)) -> List[User]:
    """Return user metadata when available.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[User]: Known user records keyed by exported identifiers.
    """
    users = service.get_users()
    return [User(**user) for user in users]


@router.get("/models", response_model=List[ModelInfo])
def list_models(service: DataService = Depends(get_data_service)) -> List[ModelInfo]:
    """Return model metadata when available.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[ModelInfo]: Known model records keyed by exported identifiers.
    """
    models = service.get_models()
    return [ModelInfo(**model) for model in models]


@router.get("/datasets/meta", response_model=DatasetMeta)
def dataset_meta(service: DataService = Depends(get_data_service)) -> DatasetMeta:
    """Return metadata about the currently loaded dataset.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        DatasetMeta: Summary statistics and provenance indicators.
    """
    return service.get_meta()


@router.post(
    "/datasets/reset",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
def reset_dataset(service: DataService = Depends(get_data_service)) -> UploadResponse:
    """Remove all stored dataset artifacts and reset metadata.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Payload describing the new dataset state.
    """
    dataset = service.reset_dataset()
    return UploadResponse(detail="Dataset reset successfully.", dataset=dataset)


@router.post(
    "/uploads/chat-export",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_chat_export(
    file: UploadFile = File(...),
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Upload a new chat export JSON file.

    Args:
        file (UploadFile): Ingested export payload from the UI.
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Dataset status after applying the upload.
    Raises:
        HTTPException: When the upload is empty or not JSON encoded.
    """
    # Validate the mimetype first; some browsers send octet-stream for JSON drag-drop uploads.
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
    """Upload a users CSV file.

    Args:
        file (UploadFile): CSV payload containing user metadata.
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Dataset status after applying the upload.
    Raises:
        HTTPException: When the upload is empty or not CSV encoded.
    """
    # Guard against incorrect uploads (e.g., XLS files) before reading them into memory.
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
    "/uploads/models",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_models_json(
    file: UploadFile = File(...),
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Upload a models JSON file.

    Args:
        file (UploadFile): JSON payload containing model metadata.
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Dataset status after applying the upload.
    Raises:
        HTTPException: When the upload is empty or not JSON encoded.
    """
    if file.content_type not in (
        "application/json",
        "text/json",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="File must be a JSON document.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    try:
        service.update_models(
            raw_bytes,
            source_label=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(detail="Models JSON uploaded successfully.", dataset=service.get_meta())


@router.post(
    "/openwebui/sync",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
def sync_openwebui(
    payload: OpenWebUISyncRequest,
    service: DataService = Depends(get_data_service),
) -> UploadResponse:
    """Fetch chats and users directly from an Open WebUI instance and store them locally.

    Args:
        payload (OpenWebUISyncRequest): Hostname and optional API key for the remote instance.
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Dataset status after syncing records.
    Raises:
        HTTPException: When validation or remote fetch fails.
    """
    try:
        dataset, stats = service.sync_from_openwebui(payload.hostname, payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return UploadResponse(
        detail="Open WebUI data synced successfully.",
        dataset=dataset,
        stats=stats,
    )


@router.get("/summaries/status")
def summary_status(service: DataService = Depends(get_data_service)) -> dict:
    """Return the current status of the summary job.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        dict: Progress metrics emitted by the summarizer worker.
    """
    return service.get_summary_status()


@router.post("/summaries/rebuild")
def rebuild_summaries(service: DataService = Depends(get_data_service)) -> dict:
    """Trigger an asynchronous rebuild of chat summaries.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        dict: Acknowledgement payload returned by the service layer.
    """
    status_state = service.rebuild_summaries()
    return {"ok": True, "status": status_state}
