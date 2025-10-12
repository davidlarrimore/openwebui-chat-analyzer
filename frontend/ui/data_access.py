from __future__ import annotations

from frontend.core.api import build_processed_data
from frontend.core.models import ProcessedData
from frontend.utils.cache import cache_data


@cache_data(show_spinner=False)
def load_processed_data(dataset_id: str) -> ProcessedData:
    """Fetch chats, messages, and users with caching per dataset."""
    return build_processed_data()

