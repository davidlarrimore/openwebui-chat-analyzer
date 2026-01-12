"""API routes for the Open WebUI Chat Analyzer backend."""

import logging
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile, status

from .auth.service import AuthService, get_auth_service
from .models import (
    AuthUserPublic,
    AdminDirectConnectSettings,
    AdminDirectConnectSettingsUpdate,
    AnonymizationSettings,
    AnonymizationSettingsUpdate,
    SummarizerSettings,
    SummarizerSettingsUpdate,
    ProviderConnection,
    ProviderConnectionsResponse,
    ProviderModel,
    ProviderModelsResponse,
    ValidateModelRequest,
    ValidateModelResponse,
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
    # Sprint 2: Multi-Metric Extraction
    MetricExtractionRequest,
    MetricExtractionResponse,
    MetricExtractionResult,
)
from .services import SUMMARY_EVENT_HISTORY_LIMIT, DataService, get_data_service
from .clients import OllamaClientError, OllamaOutOfMemoryError, get_ollama_client
from .provider_registry import get_provider_registry
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


def _resolve_openwebui_health_inputs(
    payload: Optional[OpenWebUIHealthTestRequest],
    service: DataService,
) -> Dict[str, Any]:
    """Return host/api_key for OpenWebUI health checks along with source metadata."""
    if payload and (payload.host is not None or payload.api_key is not None):
        return {
            "host": payload.host or "",
            "api_key": payload.api_key or "",
            "host_source": "request",
            "api_key_source": "request" if payload.api_key else None,
        }

    settings = service.get_direct_connect_settings()
    return {
        "host": settings.get("host", ""),
        "api_key": settings.get("api_key", ""),
        "host_source": settings.get("host_source"),
        "api_key_source": settings.get("api_key_source"),
    }


def _legacy_token_auth(
    authorization: Optional[str],
    service: DataService,
) -> AuthUserPublic:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header with Bearer token required.",
        )
    token = authorization.split(" ", 1)[1].strip()
    user_record = service.resolve_user_from_token(token)
    if user_record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    return AuthUserPublic(**user_record)


def require_authenticated_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthUserPublic:
    """Resolve the current authenticated user or raise an HTTP 401 error."""
    principal = auth_service.validate_request(request)
    if principal is not None:
        return auth_service.serialize_user(principal.user)
    LOGGER.debug("Falling back to legacy Authorization header for auth check.")
    return _legacy_token_auth(authorization, service)


def resolve_authenticated_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthUserPublic:
    """Wrapper used in FastAPI dependency injections to allow monkeypatch overrides."""
    try:
        return require_authenticated_user(
            request=request,
            authorization=authorization,
            service=service,
            auth_service=auth_service,
        )
    except TypeError:
        # Test suites may monkeypatch the dependency with a simplified zero-arg callable.
        return require_authenticated_user()  # type: ignore[misc]


def resolve_optional_authenticated_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(resolve_data_service),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[AuthUserPublic]:
    """Variant that allows anonymous access when the Authorization header is absent."""
    principal = auth_service.validate_request(request)
    if principal is not None:
        return auth_service.serialize_user(principal.user)
    if authorization is None:
        return None
    try:
        return require_authenticated_user(
            request=request,
            authorization=authorization,
            service=service,
            auth_service=auth_service,
        )
    except TypeError:
        return require_authenticated_user()  # type: ignore[misc]


@router.get("/health/ollama", tags=["health"])
def health_ollama(service: DataService = Depends(resolve_data_service)) -> Dict[str, Any]:
    """Return the health status for the Ollama service and sync models."""
    result = check_ollama_health()

    # If Ollama is healthy, sync models
    if result.status == "ok":
        try:
            service.sync_ollama_models()
        except Exception as exc:
            LOGGER.warning("Failed to sync Ollama models during health check: %s", exc)

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


@router.get("/health/openwebui", tags=["health"])
def health_openwebui_status(service: DataService = Depends(resolve_data_service)) -> Dict[str, Any]:
    """Return the health status for the configured OpenWebUI instance using stored settings."""
    inputs = _resolve_openwebui_health_inputs(payload=None, service=service)
    host = inputs["host"]
    api_key = inputs["api_key"]

    if not host or not host.strip():
        return {
            "service": "openwebui",
            "status": "error",
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "detail": "No OpenWebUI host configured. Please configure a host in settings.",
            "meta": {
                "host_source": inputs.get("host_source"),
                "api_key_source": inputs.get("api_key_source"),
                "has_api_key": bool(api_key),
            },
        }

    result = check_openwebui_health(
        host=host,
        api_key=api_key,
        interval_seconds=2.0,
        timeout_seconds=10.0,
    ).to_dict()

    meta = result.setdefault("meta", {})
    meta.update(
        {
            "host_source": inputs.get("host_source"),
            "api_key_source": inputs.get("api_key_source"),
            "has_api_key": bool(api_key),
        }
    )
    return result


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
    inputs = _resolve_openwebui_health_inputs(payload, service)
    host = inputs["host"]
    api_key = inputs["api_key"]

    # Validate that we have a host
    if not host or not host.strip():
        return {
            "service": "openwebui",
            "status": "error",
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "detail": "No OpenWebUI host configured. Please configure a host in settings.",
            "meta": {
                "host_source": inputs.get("host_source"),
                "api_key_source": inputs.get("api_key_source"),
                "has_api_key": bool(api_key),
            },
        }

    # Perform the health check
    result = check_openwebui_health(
        host=host,
        api_key=api_key,
        interval_seconds=2.0,  # Faster retries for UI responsiveness
        timeout_seconds=10.0,  # Shorter timeout for UI testing
    ).to_dict()

    meta = result.setdefault("meta", {})
    meta.update(
        {
            "host_source": inputs.get("host_source"),
            "api_key_source": inputs.get("api_key_source"),
            "has_api_key": bool(api_key),
        }
    )
    return result


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
    expose_real_names = service.should_expose_real_names()
    users = service.get_users()
    response: List[User] = []
    for user in users:
        user_id = str(user.get("user_id") or "").strip()
        if not user_id:
            continue
        real_name_raw = user.get("name")
        real_name = str(real_name_raw).strip() if isinstance(real_name_raw, str) else (
            str(real_name_raw).strip() if real_name_raw not in (None, "") else ""
        )
        pseudonym_raw = user.get("pseudonym")
        if isinstance(pseudonym_raw, str):
            pseudonym = pseudonym_raw.strip()
        elif pseudonym_raw in (None, ""):
            pseudonym = ""
        else:
            pseudonym = str(pseudonym_raw).strip()
        display_name = real_name if expose_real_names else (pseudonym or real_name)
        response.append(
            User(
                user_id=user_id,
                name=display_name or user_id,
                pseudonym=pseudonym or None,
                real_name=real_name or None,
            )
        )
    return response


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

        # If mode == "full", wipe existing chats and messages before syncing
        # Users and models are preserved and updated incrementally
        if mode == "full":
            LOGGER.info("Full sync requested - wiping existing chats and messages")
            service.wipe_chats_and_messages()
            LOGGER.info("Chats and messages wiped successfully (users and models preserved)")

        dataset, stats = service.sync_from_openwebui(
            payload.hostname,
            payload.api_key,
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        # Catch any other exceptions and log them with full traceback
        import traceback
        error_details = traceback.format_exc()
        LOGGER.error("Unexpected error during sync: %s\n%s", str(exc), error_details)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during sync: {type(exc).__name__}: {str(exc)}"
        ) from exc

    try:
        mode_label = getattr(stats, "mode", None) or "unknown"
        return UploadResponse(
            detail=f"Open WebUI data synced successfully ({mode_label} mode).",
            dataset=dataset,
            stats=stats,
        )
    except Exception as exc:
        # Catch errors during response creation and log them
        import traceback
        error_details = traceback.format_exc()
        LOGGER.error("Error creating sync response: %s\n%s", str(exc), error_details)
        raise HTTPException(
            status_code=500,
            detail=f"Sync completed but failed to create response: {type(exc).__name__}: {str(exc)}"
        ) from exc


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
    primary_model = (payload.model or "").strip()
    if not primary_model:
        try:
            primary_model = get_summary_model()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    model = (payload.model or "").strip()
    if not model:
        if not (OLLAMA_LONGFORM_MODEL or "").strip():
            raise HTTPException(
                status_code=503,
                detail="No default Ollama text generation model is configured. Provide a model in the request.",
            )
        model = (OLLAMA_LONGFORM_MODEL or "").strip()
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
    model = (payload.model or "").strip()
    if not model:
        if not (OLLAMA_LONGFORM_MODEL or "").strip():
            raise HTTPException(
                status_code=503,
                detail="No default Ollama chat model is configured. Provide a model in the request.",
            )
        model = (OLLAMA_LONGFORM_MODEL or "").strip()
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
    model = (payload.model or "").strip()
    if not model:
        if not (OLLAMA_EMBED_MODEL or "").strip():
            raise HTTPException(
                status_code=503,
                detail="No default Ollama embedding model is configured. Provide a model in the request.",
            )
        model = (OLLAMA_EMBED_MODEL or "").strip()

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
    try:
        status_state = service.rebuild_summaries()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "status": status_state}


@router.post("/summaries/run")
def run_summaries(
    mode: Literal["incremental", "full"] = Query(default="incremental"),
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> dict:
    """Trigger asynchronous summarization with an explicit mode."""
    force_resummarize = mode == "full"
    try:
        status_state = service.run_summaries(
            force_resummarize=force_resummarize,
            reason=f"manual {mode}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "mode": mode, "status": status_state}


@router.post("/summaries/cancel")
def cancel_summary_job(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> dict:
    """Cancel the currently running summarizer job.

    Args:
        service (DataService): The shared data service dependency.
    Returns:
        dict: Status payload after cancellation.
    """
    status_state = service.cancel_summary_job()
    return {"ok": True, "status": status_state}


# =========================================================================
# Sprint 2: Multi-Metric Extraction API
# =========================================================================


@router.post(
    "/metrics/extract",
    response_model=MetricExtractionResponse,
    tags=["metrics"],
)
def extract_conversation_metrics(
    request: MetricExtractionRequest,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> MetricExtractionResponse:
    """Extract specific metrics from a conversation.

    Sprint 2: Enables selective metric extraction - users can choose which
    metrics to extract (summary, outcome, tags, classification) rather than
    running all metrics for every conversation.

    This endpoint supports:
    - Selective execution (choose which metrics to extract)
    - Force re-extraction (even if metrics already exist)
    - Partial success (some metrics can fail while others succeed)

    Args:
        request: MetricExtractionRequest with chat_id and metrics list
        service: DataService instance

    Returns:
        MetricExtractionResponse with extraction results for each metric

    Example Request:
        POST /api/v1/metrics/extract
        {
            "chat_id": "chat-123",
            "metrics": ["summary", "outcome", "tags"],
            "force_reextract": false
        }

    Example Response:
        {
            "chat_id": "chat-123",
            "results": [
                {"metric_name": "summary", "success": true, "data": {"summary": "..."}},
                {"metric_name": "outcome", "success": true, "data": {"outcome": 5}},
                {"metric_name": "tags", "success": false, "error": "..."}
            ],
            "total_metrics": 3,
            "successful_metrics": 2,
            "failed_metrics": 1
        }
    """
    from backend import summarizer

    # Get chat messages
    messages = service.get_messages_for_chat(request.chat_id)
    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"Chat {request.chat_id} not found or has no messages",
        )

    # Check if metrics already exist (unless force_reextract)
    if not request.force_reextract:
        storage = service.storage
        chat_meta = storage.get_chat_metrics(request.chat_id)
        existing_metrics = chat_meta.get("metrics", {}) if chat_meta else {}

        if existing_metrics:
            # Filter out metrics that already exist
            requested_metrics = set(request.metrics)
            existing_metric_names = set(existing_metrics.keys())
            new_metrics = list(requested_metrics - existing_metric_names)

            if not new_metrics:
                # All requested metrics already exist
                extraction_metadata = chat_meta.get("extraction_metadata", {})
                return MetricExtractionResponse(
                    chat_id=request.chat_id,
                    results=[
                        MetricExtractionResult(
                            metric_name=metric,
                            success=True,
                            data={metric: existing_metrics[metric]},
                            provider=extraction_metadata.get("provider"),
                            model=extraction_metadata.get("models_used", {}).get(metric),
                        )
                        for metric in request.metrics
                        if metric in existing_metrics
                    ],
                    total_metrics=len(request.metrics),
                    successful_metrics=len(request.metrics),
                    failed_metrics=0,
                )
            # Only extract new metrics
            request.metrics = new_metrics

    # Extract metrics
    result = summarizer.extract_and_store_metrics(
        chat_id=request.chat_id,
        messages=messages,
        metrics_to_extract=request.metrics,
        storage=service.storage,
    )

    # Build response
    if not result.get("success", False):
        # Extraction failed completely
        return MetricExtractionResponse(
            chat_id=request.chat_id,
            results=[
                MetricExtractionResult(
                    metric_name=metric,
                    success=False,
                    error=result.get("error", "Unknown error"),
                    provider=None,
                    model=None,
                )
                for metric in request.metrics
            ],
            total_metrics=len(request.metrics),
            successful_metrics=0,
            failed_metrics=len(request.metrics),
        )

    # Partial or full success - retrieve stored metrics
    storage = service.storage
    chat_meta = storage.get_chat_metrics(request.chat_id) or {}
    stored_metrics = chat_meta.get("metrics", {})
    extraction_metadata = chat_meta.get("extraction_metadata", {})

    # Build result list
    results = []
    for metric in request.metrics:
        if metric in result.get("metrics_extracted", []):
            # Success
            results.append(
                MetricExtractionResult(
                    metric_name=metric,
                    success=True,
                    data={metric: stored_metrics.get(metric)},
                    provider=extraction_metadata.get("provider"),
                    model=extraction_metadata.get("models_used", {}).get(metric),
                )
            )
        else:
            # Failed
            results.append(
                MetricExtractionResult(
                    metric_name=metric,
                    success=False,
                    error=f"Failed to extract {metric}",
                    provider=extraction_metadata.get("provider"),
                    model=None,
                )
            )

    successful_count = len(result.get("metrics_extracted", []))
    failed_count = len(result.get("extraction_errors", []))

    return MetricExtractionResponse(
        chat_id=request.chat_id,
        results=results,
        total_metrics=len(request.metrics),
        successful_metrics=successful_count,
        failed_metrics=failed_count,
    )


@router.get(
    "/metrics/available",
    tags=["metrics"],
)
def get_available_metrics(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> dict:
    """Get list of available metrics that can be extracted.

    Sprint 2: Returns metadata about each available metric extractor.
    Sprint 4: Enhanced with enabled status and sprint information.

    Returns:
        Dictionary with available metrics and their metadata

    Example Response:
        {
            "metrics": [
                {
                    "name": "summary",
                    "description": "One-line conversation summary",
                    "sprint": 2,
                    "enabled_by_default": true,
                    "requires_messages": false
                },
                ...
            ]
        }
    """
    return {
        "metrics": [
            {
                "name": "summary",
                "description": "One-line conversation summary",
                "sprint": 2,
                "enabled_by_default": True,
                "requires_messages": False,
                "features": ["quality_validation"],
            },
            {
                "name": "outcome",
                "description": "Outcome score (1-5) with multi-factor evaluation",
                "sprint": 2,
                "enabled_by_default": True,
                "requires_messages": False,
                "features": ["multi_factor_scoring", "completeness", "accuracy", "helpfulness"],
            },
            {
                "name": "tags",
                "description": "Topic tags for categorization and filtering",
                "sprint": 2,
                "enabled_by_default": True,
                "requires_messages": False,
                "features": [],
            },
            {
                "name": "classification",
                "description": "Domain type and resolution status",
                "sprint": 2,
                "enabled_by_default": True,
                "requires_messages": False,
                "features": [],
            },
            {
                "name": "dropoff",
                "description": "Conversation completion and abandonment detection",
                "sprint": 3,
                "enabled_by_default": False,
                "requires_messages": True,
                "features": ["abandonment_patterns", "question_detection"],
            },
        ]
    }


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
    "/admin/settings/anonymization",
    response_model=AnonymizationSettings,
    tags=["admin"],
)
def get_admin_anonymization_settings(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> AnonymizationSettings:
    """Return anonymization mode preferences for authenticated admins."""
    settings = service.get_anonymization_settings()
    return AnonymizationSettings(**settings)


@router.put(
    "/admin/settings/anonymization",
    response_model=AnonymizationSettings,
    tags=["admin"],
)
def update_admin_anonymization_settings(
    payload: AnonymizationSettingsUpdate,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> AnonymizationSettings:
    """Update anonymization mode preferences stored in the database."""
    try:
        settings = service.update_anonymization_settings(enabled=payload.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnonymizationSettings(**settings)


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
        settings = service.update_summarizer_settings(
            model=payload.model,
            temperature=payload.temperature,
            enabled=payload.enabled,
            connection=payload.connection
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummarizerSettings(**settings)


@router.get(
    "/admin/summarizer/settings",
    response_model=SummarizerSettings,
    tags=["admin"],
)
def get_summarizer_settings_v2(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SummarizerSettings:
    """Return the active summarizer configuration (alternate path for new UI)."""
    settings = service.get_summarizer_settings()
    return SummarizerSettings(**settings)


@router.post(
    "/admin/summarizer/settings",
    response_model=SummarizerSettings,
    tags=["admin"],
)
def update_summarizer_settings_v2(
    payload: SummarizerSettingsUpdate,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> SummarizerSettings:
    """Update summarizer configuration (alternate path for new UI)."""
    try:
        settings = service.update_summarizer_settings(
            model=payload.model,
            temperature=payload.temperature,
            enabled=payload.enabled,
            connection=payload.connection
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummarizerSettings(**settings)


@router.get(
    "/admin/summarizer/statistics",
    tags=["admin"],
)
def get_summarizer_statistics(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> Dict[str, Any]:
    """Return summarizer performance statistics."""
    stats = service.get_summarizer_statistics()
    return stats


@router.post(
    "/admin/summarizer/test-connection",
    tags=["admin"],
)
def test_summarizer_connection(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
    service: DataService = Depends(resolve_data_service),
) -> Dict[str, str]:
    """Test the current summarizer connection."""
    try:
        result = service.test_summarizer_connection()
        return result
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc)
        }


@router.get(
    "/admin/summarizer/monitoring/overall",
    tags=["admin", "monitoring"],
)
def get_summarizer_overall_monitoring(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, Any]:
    """Get overall monitoring statistics (Sprint 5).

    Returns aggregated statistics across all metric types including
    success rates, latency, token usage, and retry counts.
    """
    from backend.monitoring import get_metrics_collector

    collector = get_metrics_collector()
    return collector.get_overall_stats()


@router.get(
    "/admin/summarizer/monitoring/by-metric",
    tags=["admin", "monitoring"],
)
def get_summarizer_metrics_monitoring(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, Any]:
    """Get monitoring statistics broken down by metric type (Sprint 5)."""
    from backend.monitoring import get_metrics_collector

    collector = get_metrics_collector()
    return {
        "metrics": collector.get_all_metric_stats()
    }


@router.get(
    "/admin/summarizer/monitoring/recent-failures",
    tags=["admin", "monitoring"],
)
def get_summarizer_recent_failures(
    limit: int = 50,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, Any]:
    """Get recent failure logs for debugging (Sprint 5).

    Args:
        limit: Maximum number of failures to return (default: 50, max: 200)

    Returns:
        List of recent failure logs with full details
    """
    from backend.monitoring import get_metrics_collector

    limit = min(limit, 200)  # Cap at 200
    collector = get_metrics_collector()
    return {
        "failures": collector.get_recent_failures(limit=limit)
    }


@router.get(
    "/admin/summarizer/monitoring/recent-logs",
    tags=["admin", "monitoring"],
)
def get_summarizer_recent_logs(
    limit: int = 100,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, Any]:
    """Get recent extraction logs (all attempts) for debugging (Sprint 5).

    Args:
        limit: Maximum number of logs to return (default: 100, max: 500)

    Returns:
        List of recent logs with full details
    """
    from backend.monitoring import get_metrics_collector

    limit = min(limit, 500)  # Cap at 500
    collector = get_metrics_collector()
    return {
        "logs": collector.get_recent_logs(limit=limit)
    }


@router.post(
    "/admin/summarizer/monitoring/export",
    tags=["admin", "monitoring"],
)
def export_summarizer_logs(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, str]:
    """Export all monitoring logs to a file (Sprint 5).

    Returns:
        Path to exported file
    """
    import os
    from datetime import datetime
    from backend.monitoring import get_metrics_collector

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "logs/summarizer/exports"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/export_{timestamp}.json"

        collector = get_metrics_collector()
        collector.export_logs(output_path)

        return {
            "status": "success",
            "path": output_path,
            "message": f"Logs exported to {output_path}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@router.get(
    "/admin/ollama/models",
    tags=["admin"],
)
def get_available_ollama_models(
    service: DataService = Depends(resolve_data_service),
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> Dict[str, Any]:
    """Return a list of available Ollama models with capability metadata.

    This endpoint syncs models from Ollama, tests their completion support,
    and returns the enriched model list.
    """
    try:
        # Sync models with database and registry
        sync_stats = service.sync_ollama_models()

        # Get models with capability metadata
        models = service.get_ollama_models_with_capabilities()

        return {
            "models": models,
            "sync_stats": sync_stats
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Ollama models: {str(exc)}") from exc


@router.get(
    "/admin/summarizer/connections",
    response_model=ProviderConnectionsResponse,
    tags=["admin"],
)
def get_summarizer_connections(
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> ProviderConnectionsResponse:
    """Get status of all available LLM provider connections.

    Returns the availability and configuration status of all registered providers
    (Ollama, OpenAI, Open WebUI). Unavailable providers include a reason message.
    """
    try:
        registry = get_provider_registry()
        connections_data = registry.get_available_connections()
        return ProviderConnectionsResponse(
            connections=[ProviderConnection(**conn) for conn in connections_data]
        )
    except Exception as exc:
        LOGGER.error("Failed to get provider connections: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve provider connections: {str(exc)}"
        ) from exc


@router.get(
    "/admin/summarizer/models",
    response_model=ProviderModelsResponse,
    tags=["admin"],
)
def get_summarizer_models(
    connection: str = Query(..., description="Provider type (ollama | openai | litellm | openwebui)"),
    include_unvalidated: bool = Query(True, description="Include models not yet validated"),
    auto_validate_missing: bool = Query(
        False,
        description="When true and no validated models are available, automatically revalidate discovered models.",
    ),
    force_validate: bool = Query(
        False,
        description="Force validation even if validated models already exist.",
    ),
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> ProviderModelsResponse:
    """Get models available from a specific provider.

    Discovers models from the specified provider and enriches them with validation
    status from the global registry. Only validated models are confirmed to support
    text completion/generation.
    """
    try:
        registry = get_provider_registry()
        models_data = registry.get_models_for_connection(
            connection,
            include_unvalidated,
            auto_validate_missing=auto_validate_missing,
            force_auto_validate=force_validate,
        )
        return ProviderModelsResponse(
            models=[ProviderModel(**model) for model in models_data]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error("Failed to get models for provider %s: %s", connection, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models from provider: {str(exc)}"
        ) from exc


@router.post(
    "/admin/summarizer/validate-model",
    response_model=ValidateModelResponse,
    tags=["admin"],
)
def validate_summarizer_model(
    payload: ValidateModelRequest,
    _: AuthUserPublic = Depends(resolve_authenticated_user),
) -> ValidateModelResponse:
    """Validate if a model supports text completion.

    Sends a test prompt to the model and verifies a valid response is returned.
    If successful, the model is added to the global registry of completion-capable models.
    """
    try:
        registry = get_provider_registry()
        is_valid = registry.validate_model(payload.connection, payload.model)
        return ValidateModelResponse(valid=is_valid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error(
            "Failed to validate model %s on provider %s: %s",
            payload.model,
            payload.connection,
            exc
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate model: {str(exc)}"
        ) from exc
