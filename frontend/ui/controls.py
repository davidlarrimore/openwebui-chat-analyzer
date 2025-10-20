from __future__ import annotations

import uuid
from typing import Callable, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from frontend.core.api import (
    BackendError,
    reset_dataset,
    poll_summary_status,
    sync_openwebui_dataset,
    trigger_summary_rebuild,
    upload_chat_export,
    upload_models_json,
    upload_users_csv,
)
from frontend.core.config import AppConfig
from frontend.core.models import DatasetMeta, SummaryStatus
from frontend.core.processing import determine_dataset_source
from frontend.ui.data_access import load_processed_data
from frontend.utils import state as app_state


def render_direct_connect_section(
    meta: DatasetMeta,
    *,
    dataset_ready: bool,
    config: AppConfig,
    on_refresh: Callable[[], None],
    parent: Optional[DeltaGenerator] = None,
    log_renderer: Optional[Callable[[], None]] = None,
) -> None:
    """Render the direct connect panel with full workflow."""
    source_info = determine_dataset_source(meta)
    app_state.ensure_direct_connect_state()
    app_state.set_default_openwebui_values(
        config.direct_host_default,
        config.direct_api_key_default,
    )
    render_log = log_renderer or (lambda: None)

    direct_connect_active = dataset_ready and source_info.label == "Direct Connect"
    section = parent or st
    section_title = "🔌 Direct Connect to Open WebUI"
    if direct_connect_active:
        section_title += " ✅"

    section.markdown(f"#### {section_title}")
    render_log()
    section.write(
        "Pull the latest chats and user records directly from an Open WebUI instance. "
        "Provide the base URL and an API key with sufficient permissions."
    )

    hostname_default = st.session_state.get("openwebui_hostname", config.direct_host_default)
    api_key_default = st.session_state.get("openwebui_api_key", config.direct_api_key_default)

    if "openwebui_hostname" not in st.session_state:
        st.session_state["openwebui_hostname"] = hostname_default
    if "openwebui_api_key" not in st.session_state:
        st.session_state["openwebui_api_key"] = api_key_default

    hostname = section.text_input(
        "Hostname",
        key="openwebui_hostname",
        help="Example: http://localhost:3000",
        placeholder="http://localhost:3000",
    )
    api_key = section.text_input(
        "API Key (Bearer token)",
        key="openwebui_api_key",
        type="password",
        help="Generate this token from Open WebUI settings.",
        placeholder="sk-...",
    )

    go_clicked = section.button("Load Data", key="openwebui_sync_button")

    if go_clicked:
        trimmed_host = (hostname or "").strip()
        if not trimmed_host:
            section.error("Hostname is required to connect to Open WebUI.")
            return

        app_state.reset_direct_connect_log(show=True)
        app_state.append_direct_connect_log(f"Connecting to Open WebUI at {trimmed_host}...", "🔌")
        render_log()

        try:
            sync_result = sync_openwebui_dataset(trimmed_host, api_key)
        except BackendError as exc:
            app_state.append_direct_connect_log(f"Failed to connect: {exc}", "❌")
            render_log()
            section.error(f"Failed to sync data: {exc}")
            return

        dataset_info = (sync_result.dataset or {})
        stats = sync_result.stats or {}
        chat_count = int(dataset_info.get("chat_count", 0) or 0)
        user_count = int(dataset_info.get("user_count", 0) or 0)
        message_count = int(dataset_info.get("message_count", 0) or stats.get("total_messages", 0) or 0)
        model_count = int(dataset_info.get("model_count", 0) or stats.get("total_models", 0) or 0)

        source_matched = bool(stats.get("source_matched", False))
        mode = stats.get("mode")
        normalized_host = str(stats.get("normalized_hostname") or trimmed_host)
        new_chats = int(stats.get("new_chats", 0) or 0)
        new_users = int(stats.get("new_users", 0) or 0)
        new_models = int(stats.get("new_models", 0) or 0)
        models_changed = bool(stats.get("models_changed", False))
        queued_chat_ids = stats.get("queued_chat_ids") or []
        summarizer_enqueued = stats.get("summarizer_enqueued")
        if summarizer_enqueued is None:
            summarizer_enqueued = chat_count > 0
        summarizer_enqueued = bool(summarizer_enqueued)

        def _format_count(value: int, noun: str) -> str:
            label = noun if value == 1 else f"{noun}s"
            return f"{value:,} {label}"

        if stats:
            if source_matched:
                if mode == "noop":
                    app_state.append_direct_connect_log(
                        f"Dataset already up to date for {normalized_host}; no new chats, users, or models detected.",
                        "ℹ️ ",
                    )
                else:
                    def _format_bucket(value: int, noun: str) -> str:
                        plural = "" if value == 1 else "s"
                        return f"{value:,} {noun}{plural}"

                    bucket_fragments = []
                    if new_chats:
                        bucket_fragments.append(_format_bucket(new_chats, "chat"))
                    if new_users:
                        bucket_fragments.append(_format_bucket(new_users, "user"))
                    if new_models:
                        bucket_fragments.append(_format_bucket(new_models, "model"))

                    if bucket_fragments:
                        if len(bucket_fragments) == 1:
                            counts_text = bucket_fragments[0]
                        elif len(bucket_fragments) == 2:
                            counts_text = " and ".join(bucket_fragments)
                        else:
                            counts_text = ", ".join(bucket_fragments[:-1]) + f", and {bucket_fragments[-1]}"
                        app_state.append_direct_connect_log(
                            f"Added {counts_text} from {normalized_host}.",
                            "➕",
                        )
                    elif models_changed:
                        app_state.append_direct_connect_log(
                            f"Updated model metadata from {normalized_host}.",
                            "🔄",
                        )
                    else:
                        app_state.append_direct_connect_log(
                            f"No new chats, users, or models detected for {normalized_host}.",
                            "ℹ️ ",
                        )
            else:
                app_state.append_direct_connect_log(
                    f"Loaded dataset from {normalized_host}.",
                    "📦",
                )
                app_state.append_direct_connect_log(
                    f"Retrieved {_format_count(chat_count, 'chat')} and {_format_count(user_count, 'user')}.",
                    "📥",
                )
                app_state.append_direct_connect_log(
                    f"Captured {_format_count(message_count, 'message')} and {_format_count(model_count, 'model')}.",
                    "🗂️",
                )
        else:
            app_state.append_direct_connect_log(f"Retrieved {chat_count} chats", "📥")
            app_state.append_direct_connect_log(f"Retrieved {user_count} users", "🙋")
            app_state.append_direct_connect_log(f"Retrieved {model_count} models", "🤖")

        if summarizer_enqueued:
            target_count = len(queued_chat_ids) if queued_chat_ids else (new_chats or chat_count)
            app_state.append_direct_connect_log(
                f"Creating new summaries for {_format_count(target_count, 'chat')}.",
                "🤖",
            )
        else:
            app_state.append_direct_connect_log(
                "Summarizer skipped (no new chats to process).",
                "ℹ️ ",
            )
        render_log()

        def _summary_callback(status) -> None:
            def _format_event_message(event_payload) -> str:
                payload_type = event_payload.get("type")
                if payload_type == "chat":
                    chat_id_value = event_payload.get("chat_id") or "unknown"
                    outcome = event_payload.get("outcome", "generated")
                    if outcome == "failed":
                        return f"Failed to summarize chat {chat_id_value}"
                    return f"Unexpected outcome while summarizing chat {chat_id_value}"
                return event_payload.get("message", "Processing error encountered.")

            event_payload = status.last_event or {}
            if isinstance(event_payload, dict):
                event_id = str(event_payload.get("event_id") or "")
                event_type = event_payload.get("type")
                outcome = event_payload.get("outcome")
                is_error = (event_type == "chat" and outcome == "failed") or event_type == "error"
                if is_error:
                    should_log = True
                    if event_id:
                        should_log = app_state.register_direct_connect_event(event_id)
                    if should_log:
                        log_message = event_payload.get("message") or _format_event_message(event_payload)
                        app_state.append_direct_connect_log(log_message, "⚠️")

            state = status.state
            total = status.total
            completed = status.completed
            if state == "running":
                app_state.set_direct_connect_progress(completed, total)
            elif state in {"completed", "failed", "cancelled"}:
                if total:
                    app_state.set_direct_connect_progress(completed, total)
                app_state.reset_direct_connect_progress()
            render_log()

        load_processed_data.clear()
        status = SummaryStatus(state="idle")
        if summarizer_enqueued:
            app_state.append_direct_connect_log("🧠 Building chat summaries...")
            render_log()

            with st.spinner("🧠 Building chat summaries..."):
                status = poll_summary_status(on_update=_summary_callback)

        state = status.state
        if summarizer_enqueued and state == "completed":
            app_state.append_direct_connect_log("Chat summaries rebuilt", "✅")
            st.toast("Chat summaries rebuilt.", icon="🧠")
        elif summarizer_enqueued and state == "failed":
            message = status.message or "Unknown error"
            app_state.append_direct_connect_log(f"Summary job failed: {message}", "⚠️")
            section.error(f"Summary job failed: {message}")
        elif summarizer_enqueued and state == "cancelled":
            app_state.append_direct_connect_log("Summary job cancelled (dataset changed)", "⚠️")
            section.warning("Summary job was cancelled because the dataset changed.")
        elif summarizer_enqueued and state == "running":
            app_state.append_direct_connect_log("Summaries still running in the background", "ℹ️ ")
            section.info("Summaries are still running in the background; new headlines will appear once complete.")

        app_state.append_direct_connect_log("Done", "✅")
        app_state.set_direct_connect_log_expanded(False)
        render_log()
        st.toast("Open WebUI data synced successfully!", icon="✅")
        on_refresh()


def render_upload_section(
    *,
    dataset_ready: bool,
    source_label: str,
    on_refresh: Callable[[], None],
    parent: Optional[DeltaGenerator] = None,
    log_renderer: Optional[Callable[[], None]] = None,
) -> None:
    """Render chat and user upload expanders.

    Deprecated: kept for compatibility with earlier UI flows; the Load Data page no longer calls this.
    """
    chat_upload_state = st.session_state.get("chat_file_uploader")
    chat_loaded = dataset_ready or (chat_upload_state is not None)
    file_upload_active = dataset_ready and source_label == "File Upload"
    expander_label = "📁 Upload Files"
    if file_upload_active:
        expander_label += " ✅"

    container = parent or st
    expander = container.expander(expander_label, expanded=not chat_loaded)

    with expander:
        render_log = log_renderer or (lambda: None)
        render_log()
        expander.write(
            "Load chat exports here to explore your conversations.\n\n"
            "**Download instructions**:\n"
            "- **Chats**: Admin Panel → Settings → Database → Export All Chats (All Users)\n"
            "- **Users**: Admin Panel → Settings → Database → Export Users\n"
            "- **Models**: Query the `/api/v1/models` endpoint and save the JSON response\n\n"
            "Upload files directly below. Backend defaults can still be placed in the `/data` directory."
        )

        app_state.ensure_upload_log_state()

        upload_col, users_col, models_col = expander.columns([3, 2, 2])
        with upload_col:
            upload_col.subheader("Upload Chats")
            uploaded_file = upload_col.file_uploader(
                "Select JSON file",
                type=["json"],
                help="Upload the JSON file exported from your Open WebUI instance",
                label_visibility="collapsed",
                key="chat_file_uploader",
            )
            if uploaded_file is not None:
                file_name = getattr(uploaded_file, "name", "chat export")
                app_state.reset_upload_log(show=True)
                app_state.append_upload_log(f"Uploading chat export {file_name}...", "📤")
                render_log()

                upload_failed = False
                with st.spinner("📤 Uploading chat export..."):
                    try:
                        upload_chat_export(uploaded_file)
                    except BackendError as exc:
                        upload_failed = True
                        app_state.append_upload_log(f"Upload failed: {exc}", "❌")
                        render_log()
                        upload_col.error(f"Failed to upload chat export: {exc}")
                if not upload_failed:
                    app_state.append_upload_log("Chat export uploaded. Rebuilding summaries...", "✅")
                    render_log()

                    def _format_event_message(event_payload) -> str:
                        payload_type = event_payload.get("type")
                        if payload_type == "chat":
                            chat_id_value = event_payload.get("chat_id") or "unknown"
                            outcome = event_payload.get("outcome", "generated")
                            if outcome == "failed":
                                return f"Failed to summarize chat {chat_id_value}"
                            return f"Unexpected outcome while summarizing chat {chat_id_value}"
                        return event_payload.get("message", "Progress update received.")

                    def _summary_callback(status) -> None:
                        event_payload = status.last_event or {}
                        if isinstance(event_payload, dict):
                            event_id = str(event_payload.get("event_id") or "")
                            event_type = event_payload.get("type")
                            outcome = event_payload.get("outcome")
                            is_error = (event_type == "chat" and outcome == "failed") or event_type == "error"
                            if is_error:
                                should_log = True
                                if event_id:
                                    should_log = app_state.register_upload_event(event_id)
                                if should_log:
                                    log_message = event_payload.get("message") or _format_event_message(event_payload)
                                    app_state.append_upload_log(log_message, "⚠️")

                        state = status.state
                        total = status.total
                        completed = status.completed
                        if state == "running":
                            app_state.set_upload_progress(completed, total)
                        elif state in {"completed", "failed", "cancelled"}:
                            if total:
                                app_state.set_upload_progress(completed, total)
                            app_state.reset_upload_progress()
                        render_log()

                    app_state.append_upload_log("Building chat summaries...", "🧠")
                    render_log()

                    with st.spinner("🧠 Building chat summaries..."):
                        status = poll_summary_status(on_update=_summary_callback)

                    state = status.state
                    if state == "completed":
                        app_state.append_upload_log("Chat summaries rebuilt", "✅")
                        st.toast("Chat summaries rebuilt.", icon="🧠")
                    elif state == "failed":
                        message = status.message or "Unknown error"
                        app_state.append_upload_log(f"Summary job failed: {message}", "⚠️")
                        upload_col.error(f"Summary job failed: {message}")
                    elif state == "cancelled":
                        app_state.append_upload_log("Summary job cancelled (dataset changed)", "⚠️")
                        upload_col.warning("Summary job was cancelled because the dataset changed.")
                    else:
                        app_state.append_upload_log("No active summarizer job detected.", "ℹ️ ")

                    app_state.append_upload_log("Done", "✅")
                    app_state.set_upload_log_expanded(False)
                    render_log()

                    load_processed_data.clear()
                    st.session_state.pop("chat_file_uploader", None)
                    st.toast("Chat export uploaded successfully!", icon="✅")
                    on_refresh()
        with users_col:
            users_col.subheader("Upload Users")
            uploaded_users_file = users_col.file_uploader(
                "Optional users CSV",
                type=["csv"],
                help="Upload a CSV with `user_id` and `name` columns to map chats to user names",
                label_visibility="collapsed",
                key="users_csv_uploader",
            )
            if uploaded_users_file is not None:
                file_name = getattr(uploaded_users_file, "name", "users.csv")
                app_state.reset_upload_log(show=True)
                app_state.append_upload_log(f"Uploading users CSV {file_name}...", "📤")
                render_log()

                upload_failed = False
                with st.spinner("📤 Uploading users CSV..."):
                    try:
                        upload_users_csv(uploaded_users_file)
                    except BackendError as exc:
                        upload_failed = True
                        app_state.append_upload_log(f"Upload failed: {exc}", "❌")
                        render_log()
                        users_col.error(f"Failed to upload users CSV: {exc}")

                if not upload_failed:
                    app_state.append_upload_log("Users CSV imported.", "✅")
                    render_log()
                    load_processed_data.clear()
                    st.session_state.pop("users_csv_uploader", None)
                    app_state.append_upload_log("Done", "✅")
                    app_state.set_upload_log_expanded(False)
                    render_log()
                    st.toast("Users CSV uploaded successfully!", icon="✅")
                    on_refresh()

        with models_col:
            models_col.subheader("Upload Models")
            uploaded_models_file = models_col.file_uploader(
                "Optional models JSON",
                type=["json"],
                help="Upload JSON from the `/api/v1/models` endpoint to map model IDs to friendly names",
                label_visibility="collapsed",
                key="models_json_uploader",
            )
            if uploaded_models_file is not None:
                file_name = getattr(uploaded_models_file, "name", "models.json")
                app_state.reset_upload_log(show=True)
                app_state.append_upload_log(f"Uploading model catalog {file_name}...", "📦")
                render_log()

                upload_failed = False
                with st.spinner("📦 Uploading model catalog..."):
                    try:
                        upload_models_json(uploaded_models_file)
                    except BackendError as exc:
                        upload_failed = True
                        app_state.append_upload_log(f"Upload failed: {exc}", "❌")
                        render_log()
                        models_col.error(f"Failed to upload models JSON: {exc}")

                if not upload_failed:
                    app_state.append_upload_log("Model catalog imported.", "✅")
                    app_state.append_upload_log("Updating cached model names...", "🛠️")
                    render_log()
                    load_processed_data.clear()
                    st.session_state.pop("models_json_uploader", None)
                    app_state.append_upload_log("Done", "✅")
                    app_state.set_upload_log_expanded(False)
                    render_log()
                    st.toast("Models JSON uploaded successfully!", icon="✅")
                    on_refresh()


def render_admin_section(
    *,
    dataset: DatasetMeta,
    on_refresh: Callable[[], None],
    parent: Optional[DeltaGenerator] = None,
    log_renderer: Optional[Callable[[], None]] = None,
) -> None:
    """Render administrative actions for managing the local dataset."""
    section = parent or st
    render_log = log_renderer or (lambda: None)
    panel = section.container(border=True)
    panel.markdown(
        f"""
        <div style="
            background-color:#f3f4f6;
            color:#374151;
            font-weight:600;
            font-size:1.0rem;
            padding:0.65rem 1.1rem;
            border-radius:8px;
            margin-bottom:0.9rem;
            box-shadow:inset 0 0 0 1px rgba(191,219,254,0.7);
        ">
            🛡️ Admin Tools
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_log()
    panel.warning(
        "These actions affect all loaded data. Proceed only if you understand the consequences.",
        icon="⚠️",
    )

    delete_disabled = dataset.chat_count == 0 and dataset.user_count == 0 and dataset.model_count == 0
    rerun_disabled = dataset.chat_count == 0

    button_col, rerun_col = panel.columns([1, 1])

    delete_clicked = button_col.button(
        "🗑️ Delete all data",
        key="admin_delete_all_button",
        type="primary",
        help="Remove all chats, messages, users, models, and reset local app metadata.",
        disabled=delete_disabled,
    )
    rerun_clicked = rerun_col.button(
        "🔁 Rerun summaries",
        key="admin_rerun_summaries_button",
        type="secondary",
        help="Rebuild chat summaries using the backend summarizer service.",
        disabled=rerun_disabled,
    )

    panel.markdown(
        """
        <script>
        (function attachConfirmWatcher() {
            const doc = window.document;
            if (!doc || !doc.body) { return; }

            const confirmText = 'This will permanently delete all chats, messages, users, models, and reset app.json. Are you sure?';

            function bindOnce() {
                const buttons = Array.from(doc.querySelectorAll('button'));
                const target = buttons.find((btn) => (btn.innerText || '').trim().includes('Delete all data'));
                if (!target || target.dataset.confirmBound === 'true') {
                    return;
                }
                target.dataset.confirmBound = 'true';
                target.addEventListener('click', function(event) {
                    const confirmed = window.confirm(confirmText);
                    if (!confirmed) {
                        event.stopImmediatePropagation();
                        event.preventDefault();
                    }
                }, true);
            }

            const observer = new MutationObserver(bindOnce);
            observer.observe(doc.body, { childList: true, subtree: true });
            bindOnce();
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    if delete_clicked:
        app_state.reset_admin_log(show=True)
        app_state.append_admin_log("Deleting chats, messages, users, and models...", "🗑️")
        render_log()

        try:
            result = reset_dataset()
        except BackendError as exc:
            app_state.append_admin_log(f"Failed to delete data: {exc}", "❌")
            render_log()
            panel.error(f"Failed to delete all data: {exc}")
        else:
            load_processed_data.clear()
            meta_payload = result.dataset or {}
            try:
                new_meta = DatasetMeta.from_dict(meta_payload)
            except Exception:
                new_meta = dataset
            app_state.set_dataset_meta(new_meta)
            app_state.append_admin_log("Dataset reset complete.", "✅")
            app_state.set_admin_log_expanded(False)
            render_log()
            st.toast("All analyzer data deleted.", icon="🗑️")
            on_refresh()

    if rerun_clicked:
        app_state.reset_admin_log(show=True)
        app_state.append_admin_log("Queuing manual summary rebuild...", "🔁")
        render_log()

        def _format_event_message(event_payload) -> str:
            payload_type = event_payload.get("type")
            if payload_type == "start":
                return event_payload.get("message", "")
            if payload_type == "chat":
                chat_id_value = event_payload.get("chat_id") or "unknown"
                outcome = event_payload.get("outcome", "generated")
                if outcome == "failed":
                    return f"Failed to summarize chat {chat_id_value}"
                return f"Unexpected outcome while summarizing chat {chat_id_value}"
            return event_payload.get("message", "Progress update received.")

        def _summary_callback(status) -> None:
            event_payload = status.last_event or {}
            if isinstance(event_payload, dict):
                event_id = str(event_payload.get("event_id") or "")
                event_type = event_payload.get("type")
                outcome = event_payload.get("outcome")
                is_error = (event_type == "chat" and outcome == "failed") or event_type == "error"
                if is_error:
                    should_log = True
                    if event_id:
                        should_log = app_state.register_admin_event(event_id)
                    if should_log:
                        log_message = event_payload.get("message") or _format_event_message(event_payload)
                        app_state.append_admin_log(log_message, "⚠️")

            state = status.state
            total = status.total
            completed = status.completed
            if state == "running":
                app_state.set_admin_progress(completed, total)
            elif state in {"completed", "failed", "cancelled"}:
                if total:
                    app_state.set_admin_progress(completed, total)
                app_state.reset_admin_progress()
            render_log()

        try:
            initial_status = trigger_summary_rebuild()
        except BackendError as exc:
            app_state.append_admin_log(f"Failed to start summarizer: {exc}", "❌")
            render_log()
            panel.error(f"Unable to start summary rebuild: {exc}")
            return

        if initial_status.last_event:
            _summary_callback(initial_status)

        state = initial_status.state
        if state in {None, "idle"} and dataset.chat_count == 0:
            message = initial_status.message or "No chats available to summarize."
            app_state.append_admin_log(message, "ℹ️ ")
            render_log()
            panel.info(message)
            return
        if state == "failed":
            message = initial_status.message or "Summary job failed to start."
            app_state.append_admin_log(f"Summary job failed: {message}", "⚠️")
            render_log()
            panel.error(message)
            return

        app_state.append_admin_log("Rebuilding chat summaries...", "🧠")
        render_log()

        with st.spinner("🧠 Rebuilding chat summaries..."):
            final_status = poll_summary_status(on_update=_summary_callback)

        final_state = final_status.state
        if final_state == "completed":
            app_state.append_admin_log("Chat summaries rebuilt", "✅")
            app_state.set_admin_log_expanded(False)
            load_processed_data.clear()
            render_log()
            st.toast("Chat summaries rebuilt.", icon="🧠")
            on_refresh()
        elif final_state == "failed":
            message = final_status.message or "Summary job failed."
            app_state.append_admin_log(f"Summary job failed: {message}", "⚠️")
            render_log()
            panel.error(f"Summary job failed: {message}")
        elif final_state == "cancelled":
            app_state.append_admin_log("Summary job cancelled (dataset changed).", "⚠️")
            render_log()
            panel.warning("Summary job was cancelled because the dataset changed.")
        else:
            app_state.append_admin_log("Summaries still running in the background.", "ℹ️ ")
            render_log()
            panel.info("Summaries are still running in the background; refresh later for updates.")
