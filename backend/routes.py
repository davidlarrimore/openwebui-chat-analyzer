"""API routes for the Open WebUI Chat Analyzer backend."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status

from .models import (
    AuthLoginRequest,
    AuthLoginResponse,
    AuthStatus,
    AuthUserCreate,
    AuthUserPublic,
    AdminDirectConnectSettings,
    AdminDirectConnectSettingsUpdate,
    SummarizerSettings,
    SummarizerSettingsUpdate,
    OpenWebUIHealthTestRequest,
    Chat,
    DatasetMeta,
    Message,
    IngestLogEntry,
    ModelInfo,
    OpenWebUISyncRequest,
    ProcessLogEvent,
    ProcessLogsResponse,
    SyncSchedulerConfig,
    SyncSchedulerUpdate,
    SyncStatusResponse,
    UploadResponse,
    User,
    GenAIGenerateRequest,
    GenAIGenerateResponse,
    GenAIChatRequest,
    GenAIChatResponse,
    GenAIEmbedRequest,
    GenAIEmbedResponse,
    GenAISummarizeRequest,
    GenAISummarizeResponse,
    GenAIMessage,
)
from .services import SUMMARY_EVENT_HISTORY_LIMIT, DataService, get_data_service
from .clients import OllamaClientError, OllamaOutOfMemoryError, get_ollama_client
from .config import (
    OLLAMA_DEFAULT_TEMPERATURE,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LONGFORM_MODEL,
    OLLAMA_KEEP_ALIVE,
)
from .summarizer import (
    _HEADLINE_SYS,
    _HEADLINE_USER_TMPL,
    _trim_one_line,
    get_summary_model,
    get_summary_fallback_model,
)
from .health import check_backend_health, check_database_health, check_ollama_health, check_openwebui_health

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["data"])


def resolve_data_service() -> DataService:
    """Wrapper to allow monkeypatching of the shared data service dependency."""
    return get_data_service()


def require_authenticated_user(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
) -> AuthUserPublic:
    """Resolve the current authenticated user or raise an HTTP 401 error."""
    if not authorization or not authorization.lower().startswith("bearer "):
        LOGGER.warning("Authorization header missing or malformed during auth check.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header with Bearer token required.",
        )
    token = authorization.split(" ", 1)[1].strip()
    user_record = service.resolve_user_from_token(token)
    if user_record is None:
        LOGGER.warning("Access token validation failed during auth check.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    LOGGER.debug("Authenticated user %s via access token.", user_record["username"])
    return AuthUserPublic(**user_record)


def resolve_authenticated_user(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
) -> AuthUserPublic:
    """Wrapper used in FastAPI dependency injections to allow monkeypatch overrides."""
    try:
        return require_authenticated_user(authorization=authorization, service=service)
    except TypeError:
        # Test suites may monkeypatch the dependency with a simplified zero-arg callable.
        return require_authenticated_user()  # type: ignore[misc]


def resolve_optional_authenticated_user(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
) -> Optional[AuthUserPublic]:
    """Variant that allows anonymous access when the Authorization header is absent."""
    if authorization is None:
        return None
    try:
        return require_authenticated_user(authorization=authorization, service=service)
    except TypeError:
        return require_authenticated_user()  # type: ignore[misc]


@router.get("/health/ollama", tags=["health"])
def health_ollama() -> Dict[str, Any]:
    """Return the health status for the Ollama service."""
    result = check_ollama_health()
    return result.to_dict()


@router.get("/health/database", tags=["health"])
def health_database() -> Dict[str, Any]:
    """Return the health status for the database connection."""
    result = check_database_health()
    return result.to_dict()


@router.get("/health/backend", tags=["health"])
def health_backend() -> Dict[str, Any]:
    """Return the health status for the backend API."""
    result = check_backend_health()
    return result.to_dict()


@router.post("/health/openwebui", tags=["health"])
def health_openwebui(
    payload: OpenWebUIHealthTestRequest,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> Dict[str, Any]:
    """Test connectivity to an OpenWebUI instance.

    This endpoint tests reachability and authentication against the specified
    OpenWebUI host. If no host/api_key are provided in the request, it uses
    the stored Direct Connect settings.

    Returns:
        Health check result with status, connection metadata, and any error details.
    """
    # Get host and API key from request or fall back to stored settings
    if payload.host is not None or payload.api_key is not None:
        # Use explicitly provided values
        host = payload.host or ""
        api_key = payload.api_key or ""
    else:
        # Fall back to stored settings
        settings = service.get_direct_connect_settings()
        host = settings.get("host", "")
        api_key = settings.get("api_key", "")

    # Validate that we have a host
    if not host or not host.strip():
        return {
            "service": "openwebui",
            "status": "error",
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "detail": "No OpenWebUI host configured. Please configure a host in settings.",
        }

    # Perform the health check
    result = check_openwebui_health(
        host=host,
        api_key=api_key,
        interval_seconds=2.0,  # Faster retries for UI responsiveness
        timeout_seconds=10.0,  # Shorter timeout for UI testing
    )

    return result.to_dict()


@router.get("/chats", response_model=List[Chat])
def list_chats(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> List[Chat]:
    """Return all chat metadata.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[Chat]: Chats converted into the API schema.
    """
    chats = service.get_chats()
    return [Chat(**chat) for chat in chats]


@router.get("/messages", response_model=List[Message])
def list_messages(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> List[Message]:
    """Return the flattened list of chat messages.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[Message]: Serialized messages compliant with the API schema.
    """
    messages = service.get_messages()
    return [Message(**message) for message in messages]


@router.get("/users", response_model=List[User])
def list_users(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> List[User]:
    """Return user metadata when available.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[User]: Known user records keyed by exported identifiers.
    """
    users = service.get_users()
    return [User(**user) for user in users]


@router.get("/models", response_model=List[ModelInfo])
def list_models(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> List[ModelInfo]:
    """Return model metadata when available.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        List[ModelInfo]: Known model records keyed by exported identifiers.
    """
    models = service.get_models()
    return [ModelInfo(**model) for model in models]


@router.get("/datasets/meta", response_model=DatasetMeta)
def dataset_meta(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> DatasetMeta:
    """Return metadata about the currently loaded dataset.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        DatasetMeta: Summary statistics and provenance indicators.
    """
    return service.get_meta()


@router.get("/logs/ingest", response_model=List[IngestLogEntry])
def list_ingest_logs(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    limit: int = 50,
    service: DataService = Depends(resolve_data_service),
) -> List[IngestLogEntry]:
    """Return recent ingest log entries."""
    sanitized_limit = max(1, min(limit, 500))
    logs = service.get_ingest_logs(limit=sanitized_limit)
    return [IngestLogEntry(**log) for log in logs]


@router.post(
    "/datasets/reset",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
def reset_dataset(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> UploadResponse:
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
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
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
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
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
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
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


@router.get("/sync/status", response_model=SyncStatusResponse)
def get_sync_status(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SyncStatusResponse:
    """Get the current sync status and watermark information.

    Returns:
        SyncStatusResponse: Last sync timestamp, watermark, and recommended mode.
    """
    return service.get_sync_status()


@router.get("/logs", response_model=ProcessLogsResponse)
def get_process_logs(
    job_id: Optional[str] = Query(None, description="Filter logs by job ID"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of logs to return"),
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> ProcessLogsResponse:
    """Retrieve recent process log events from the in-memory buffer.

    Args:
        job_id: Optional job ID to filter logs by specific operation
        limit: Maximum number of log events to return (1-500, default 100)
        service: The shared data service dependency

    Returns:
        ProcessLogsResponse: List of log events and total count
    """
    logs_data = service.get_process_logs(job_id=job_id, limit=limit)

    # Convert to ProcessLogEvent models
    log_events = [
        ProcessLogEvent(
            timestamp=log["timestamp"],
            level=log["level"],
            job_id=log.get("job_id"),
            phase=log["phase"],
            message=log["message"],
            details=log.get("details"),
        )
        for log in logs_data
    ]

    return ProcessLogsResponse(
        logs=log_events,
        total=len(log_events),
    )


@router.post(
    "/openwebui/sync",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
def sync_openwebui(
    payload: OpenWebUISyncRequest,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> UploadResponse:
    """Fetch chats and users directly from an Open WebUI instance and store them locally.

    Args:
        payload (OpenWebUISyncRequest): Hostname, optional API key, and sync mode.
        service (DataService): The shared data service dependency.
    Returns:
        UploadResponse: Dataset status after syncing records.
    Raises:
        HTTPException: When validation or remote fetch fails.
    """
    try:
        # Note: sync_from_openwebui automatically determines full vs incremental
        # based on source matching, but we respect the user's mode preference
        # by potentially clearing data first
        mode = payload.mode
        if mode is None:
            # Default to incremental if we have data, full if empty
            sync_status = service.get_sync_status()
            mode = sync_status.recommended_mode

        # TODO: If mode == "full", consider clearing existing data before sync
        # For now, sync_from_openwebui handles this via source matching logic

        dataset, stats = service.sync_from_openwebui(
            payload.hostname,
            payload.api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    mode_label = getattr(stats, "mode", None) or "unknown"
    return UploadResponse(
        detail=f"Open WebUI data synced successfully ({mode_label} mode).",
        dataset=dataset,
        stats=stats,
    )


@router.get("/sync/scheduler", response_model=SyncSchedulerConfig)
def get_sync_scheduler(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SyncSchedulerConfig:
    """Get current sync scheduler configuration and state.

    Returns:
        SyncSchedulerConfig: Scheduler enabled state, interval, and timestamps.
    """
    config = service.get_scheduler_config()
    return SyncSchedulerConfig(**config)


@router.post("/sync/scheduler", response_model=SyncSchedulerConfig)
def update_sync_scheduler(
    payload: SyncSchedulerUpdate,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SyncSchedulerConfig:
    """Update sync scheduler configuration.

    Args:
        payload: Scheduler configuration updates (enabled, interval_minutes).

    Returns:
        SyncSchedulerConfig: Updated scheduler configuration.

    Raises:
        HTTPException: When validation fails (e.g., invalid interval).
    """
    try:
        config = service.update_scheduler_config(
            enabled=payload.enabled,
            interval_minutes=payload.interval_minutes,
        )
        return SyncSchedulerConfig(**config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/genai/summarize",
    response_model=GenAISummarizeResponse,
    tags=["genai"],
)
def generate_summary(
    payload: GenAISummarizeRequest,
    _: Optional[AuthUserPublic] = Depends(resolve_optional_authenticated_user),
) -> GenAISummarizeResponse:
    """Generate a concise summary using the Ollama service."""
    context = payload.context.strip()
    primary_model = (payload.model or "").strip() or get_summary_model()
    fallback_model = None
    if not payload.model:
        fallback_candidate = get_summary_fallback_model()
        if fallback_candidate and fallback_candidate != primary_model:
            fallback_model = fallback_candidate
    temperature = (
        payload.temperature if payload.temperature is not None else OLLAMA_DEFAULT_TEMPERATURE
    )
    # Use max_chars if provided for token prediction, otherwise use reasonable default
    num_predict = min(48, max(32, (payload.max_chars or 256) // 8))

    if not context:
        return GenAISummarizeResponse(summary="", model=primary_model)

    client = get_ollama_client()
    prompt = _HEADLINE_USER_TMPL.format(ctx=context)
    options = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": 1024,
    }

    active_model = primary_model
    try:
        result = client.generate(
            prompt=prompt,
            model=active_model,
            system=_HEADLINE_SYS,
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except OllamaOutOfMemoryError as exc:
        if not fallback_model:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        LOGGER.warning(
            "Primary Ollama model %s ran out of memory; retrying with fallback model %s.",
            active_model,
            fallback_model,
            exc_info=True,
        )
        active_model = fallback_model
        try:
            result = client.generate(
                prompt=prompt,
                model=active_model,
                system=_HEADLINE_SYS,
                options=options,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
        except OllamaClientError as fallback_exc:
            raise HTTPException(status_code=502, detail=str(fallback_exc)) from fallback_exc
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    raw_summary = result.response or ""
    primary_line = raw_summary.splitlines()[0] if raw_summary else ""
    normalized = " ".join(primary_line.strip().split())
    # Apply client-requested max_chars truncation if specified
    if payload.max_chars and normalized and len(normalized) > payload.max_chars:
        summary = normalized[: payload.max_chars - 1].rstrip() + "…"
    else:
        summary = normalized
    return GenAISummarizeResponse(summary=summary, model=result.model or active_model)


@router.post(
    "/genai/generate",
    response_model=GenAIGenerateResponse,
    tags=["genai"],
)
def generate_text(
    payload: GenAIGenerateRequest,
    _: Optional[AuthUserPublic] = Depends(resolve_optional_authenticated_user),
) -> GenAIGenerateResponse:
    """Run a single-prompt generation request against Ollama."""
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    client = get_ollama_client()
    model = payload.model or OLLAMA_LONGFORM_MODEL
    options = dict(payload.options or {})
    options.setdefault("temperature", OLLAMA_DEFAULT_TEMPERATURE)

    try:
        result = client.generate(
            prompt=payload.prompt,
            model=model,
            system=payload.system,
            options=options or None,
        )
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GenAIGenerateResponse(text=result.response, model=result.model or model, raw=result.raw)


@router.post(
    "/genai/chat",
    response_model=GenAIChatResponse,
    tags=["genai"],
)
def chat(
    payload: GenAIChatRequest,
    _: Optional[AuthUserPublic] = Depends(resolve_optional_authenticated_user),
) -> GenAIChatResponse:
    """Run a multi-turn chat completion against Ollama."""
    if not payload.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    client = get_ollama_client()
    model = payload.model or OLLAMA_LONGFORM_MODEL
    options = dict(payload.options or {})
    options.setdefault("temperature", OLLAMA_DEFAULT_TEMPERATURE)

    try:
        result = client.chat(
            messages=[message.dict() for message in payload.messages],
            model=model,
            options=options or None,
        )
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    assistant_message = GenAIMessage(**result.message)
    return GenAIChatResponse(message=assistant_message, model=result.model or model, raw=result.raw)


@router.post(
    "/genai/embed",
    response_model=GenAIEmbedResponse,
    tags=["genai"],
)
def embed(
    payload: GenAIEmbedRequest,
    _: Optional[AuthUserPublic] = Depends(resolve_optional_authenticated_user),
) -> GenAIEmbedResponse:
    """Generate embeddings for a list of strings."""
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty.")

    client = get_ollama_client()
    model = payload.model or OLLAMA_EMBED_MODEL

    try:
        result = client.embed(inputs=payload.inputs, model=model)
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GenAIEmbedResponse(
        embeddings=result.embeddings,
        model=result.model or model,
        raw=result.raw,
    )


@router.get(
    "/genai/models",
    response_model=List[Dict[str, Any]],
    tags=["genai"],
)
def list_genai_models(
    _: Optional[AuthUserPublic] = Depends(resolve_optional_authenticated_user),
) -> List[Dict[str, Any]]:
    """Return the set of models currently available within Ollama."""
    client = get_ollama_client()
    try:
        return client.list_models()
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/summaries/status")
def summary_status(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> dict:
    """Return the current status of the summary job.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        dict: Progress metrics emitted by the summarizer worker.
    """
    return service.get_summary_status()


@router.get("/summaries/events")
def summary_events(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    after: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=SUMMARY_EVENT_HISTORY_LIMIT),
    service: DataService = Depends(resolve_data_service),
) -> dict:
    """Return queued summarizer events emitted by the worker."""
    return service.get_summary_events(after=after, limit=limit)


@router.post("/summaries/rebuild")
def rebuild_summaries(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> dict:
    """Trigger an asynchronous rebuild of chat summaries.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        dict: Acknowledgement payload returned by the service layer.
    """
    status_state = service.rebuild_summaries()
    return {"ok": True, "status": status_state}


@router.get("/auth/status", response_model=AuthStatus, tags=["auth"])
def auth_status(service: DataService = Depends(resolve_data_service)) -> AuthStatus:
    """Report whether any authentication users have been created."""
    return AuthStatus(has_users=service.has_auth_users())


@router.post(
    "/auth/bootstrap",
    response_model=AuthLoginResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def bootstrap_first_user(
    payload: AuthUserCreate,
    service: DataService = Depends(resolve_data_service),
) -> AuthLoginResponse:
    """Create the initial authentication user when none exist yet."""
    if service.has_auth_users():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication users already exist.",
        )
    try:
        created = service.create_auth_user(payload.username, payload.password, name=payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    token = service.issue_access_token(created["username"])
    user_public = AuthUserPublic(
        id=created["username"],
        username=created["username"],
        email=created["username"],
        name=created["name"],
    )
    return AuthLoginResponse(access_token=token, token_type="bearer", user=user_public)


@router.post("/auth/login", response_model=AuthLoginResponse, tags=["auth"])
def login(
    payload: AuthLoginRequest,
    service: DataService = Depends(resolve_data_service),
) -> AuthLoginResponse:
    """Authenticate user credentials and return a bearer token."""
    LOGGER.info("Login attempt for user '%s'.", payload.username)
    user_record = service.authenticate_credentials(payload.username, payload.password)
    if user_record is None:
        LOGGER.warning("Failed login attempt for user '%s'.", payload.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token = service.issue_access_token(user_record["username"])
    user_public = AuthUserPublic(**user_record)
    LOGGER.info("Issued new access token for user '%s'.", user_record["username"])
    return AuthLoginResponse(access_token=token, token_type="bearer", user=user_public)


@router.post("/auth/logout", tags=["auth"], status_code=status.HTTP_200_OK)
def logout(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
) -> Dict[str, Any]:
    """Revoke the current access token, logging out the user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        LOGGER.info("Logout requested without bearer token; nothing to revoke.")
        return {"ok": True, "detail": "No token provided."}
    token = authorization.split(" ", 1)[1].strip()
    revoked = service.revoke_token(token)
    LOGGER.info("Processed logout; token revoked=%s.", revoked)
    return {"ok": True, "revoked": revoked}


@router.get("/users/me", response_model=AuthUserPublic, tags=["auth"])
def current_user(user: AuthUserPublic = Depends(resolve_authenticated_user)) -> AuthUserPublic:
    """Resolve the current authenticated user from an Authorization header."""
    return user


@router.get(
    "/admin/settings/direct-connect",
    response_model=AdminDirectConnectSettings,
    tags=["admin"],
)
def get_admin_direct_connect_settings(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> AdminDirectConnectSettings:
    """Return stored Direct Connect defaults for authenticated admins."""
    settings = service.get_direct_connect_settings()
    return AdminDirectConnectSettings(**settings)


@router.put(
    "/admin/settings/direct-connect",
    response_model=AdminDirectConnectSettings,
    tags=["admin"],
)
def update_admin_direct_connect_settings(
    payload: AdminDirectConnectSettingsUpdate,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> AdminDirectConnectSettings:
    """Update Direct Connect defaults stored in the database."""
    settings = service.update_direct_connect_settings(
        host=payload.host,
        api_key=payload.api_key,
    )
    return AdminDirectConnectSettings(**settings)


@router.get(
    "/admin/settings/summarizer",
    response_model=SummarizerSettings,
    tags=["admin"],
)
def get_admin_summarizer_settings(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SummarizerSettings:
    """Return the active summarizer configuration for authenticated admins."""
    settings = service.get_summarizer_settings()
    return SummarizerSettings(**settings)


@router.put(
    "/admin/settings/summarizer",
    response_model=SummarizerSettings,
    tags=["admin"],
)
def update_admin_summarizer_settings(
    payload: SummarizerSettingsUpdate,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SummarizerSettings:
    """Persist summarizer configuration updates."""
    try:
        settings = service.update_summarizer_settings(model=payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummarizerSettings(**settings)
