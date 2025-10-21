"""API routes for the Open WebUI Chat Analyzer backend."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status

from .models import (
    AuthLoginRequest,
    AuthLoginResponse,
    AuthStatus,
    AuthUserCreate,
    AuthUserPublic,
    AdminDirectConnectSettings,
    AdminDirectConnectSettingsUpdate,
    Chat,
    DatasetMeta,
    Message,
    IngestLogEntry,
    ModelInfo,
    OpenWebUISyncRequest,
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
from .services import DataService, get_data_service
from .clients import OllamaClientError, OllamaOutOfMemoryError, get_ollama_client
from .config import (
    OLLAMA_DEFAULT_TEMPERATURE,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LONGFORM_MODEL,
    OLLAMA_SUMMARY_MODEL,
    OLLAMA_SUMMARY_FALLBACK_MODEL,
)
from .summarizer import MAX_CHARS, _HEADLINE_SYS, _HEADLINE_USER_TMPL, _trim_one_line
from .health import check_database_health, check_ollama_health

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["data"])


def require_authenticated_user(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    service: DataService = Depends(get_data_service),
) -> AuthUserPublic:
    """Resolve the current authenticated user or raise an HTTP 401 error."""
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


@router.get("/logs/ingest", response_model=List[IngestLogEntry])
def list_ingest_logs(
    limit: int = 50,
    service: DataService = Depends(get_data_service),
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


@router.post(
    "/genai/summarize",
    response_model=GenAISummarizeResponse,
    tags=["genai"],
)
def generate_summary(payload: GenAISummarizeRequest) -> GenAISummarizeResponse:
    """Generate a concise summary using the Ollama service."""
    context = payload.context.strip()
    primary_model = payload.model or OLLAMA_SUMMARY_MODEL
    fallback_model = (
        OLLAMA_SUMMARY_FALLBACK_MODEL
        if not payload.model
        and OLLAMA_SUMMARY_FALLBACK_MODEL
        and OLLAMA_SUMMARY_FALLBACK_MODEL != primary_model
        else None
    )
    char_limit = payload.max_chars or MAX_CHARS
    temperature = (
        payload.temperature if payload.temperature is not None else OLLAMA_DEFAULT_TEMPERATURE
    )

    if not context:
        return GenAISummarizeResponse(summary="", model=primary_model)

    client = get_ollama_client()
    prompt = _HEADLINE_USER_TMPL.format(ctx=context)
    options = {
        "temperature": temperature,
        "num_predict": max(128, min(char_limit * 2, 512)),
    }

    active_model = primary_model
    try:
        result = client.generate(
            prompt=prompt,
            model=active_model,
            system=_HEADLINE_SYS,
            options=options,
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
            )
        except OllamaClientError as fallback_exc:
            raise HTTPException(status_code=502, detail=str(fallback_exc)) from fallback_exc
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    summary = _trim_one_line(result.response, char_limit)
    return GenAISummarizeResponse(summary=summary, model=result.model or active_model)


@router.post(
    "/genai/generate",
    response_model=GenAIGenerateResponse,
    tags=["genai"],
)
def generate_text(payload: GenAIGenerateRequest) -> GenAIGenerateResponse:
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
def chat(payload: GenAIChatRequest) -> GenAIChatResponse:
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
def embed(payload: GenAIEmbedRequest) -> GenAIEmbedResponse:
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
def list_genai_models() -> List[Dict[str, Any]]:
    """Return the set of models currently available within Ollama."""
    client = get_ollama_client()
    try:
        return client.list_models()
    except OllamaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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


@router.get("/auth/status", response_model=AuthStatus, tags=["auth"])
def auth_status(service: DataService = Depends(get_data_service)) -> AuthStatus:
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
    service: DataService = Depends(get_data_service),
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
    service: DataService = Depends(get_data_service),
) -> AuthLoginResponse:
    """Authenticate user credentials and return a bearer token."""
    user_record = service.authenticate_credentials(payload.username, payload.password)
    if user_record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token = service.issue_access_token(user_record["username"])
    user_public = AuthUserPublic(**user_record)
    return AuthLoginResponse(access_token=token, token_type="bearer", user=user_public)


@router.get("/users/me", response_model=AuthUserPublic, tags=["auth"])
def current_user(user: AuthUserPublic = Depends(require_authenticated_user)) -> AuthUserPublic:
    """Resolve the current authenticated user from an Authorization header."""
    return user


@router.get(
    "/admin/settings/direct-connect",
    response_model=AdminDirectConnectSettings,
    tags=["admin"],
)
def get_admin_direct_connect_settings(
    _: AuthUserPublic = Depends(require_authenticated_user),
    service: DataService = Depends(get_data_service),
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
    _: AuthUserPublic = Depends(require_authenticated_user),
    service: DataService = Depends(get_data_service),
) -> AdminDirectConnectSettings:
    """Update Direct Connect defaults stored in the database."""
    settings = service.update_direct_connect_settings(
        host=payload.host,
        api_key=payload.api_key,
    )
    return AdminDirectConnectSettings(**settings)
