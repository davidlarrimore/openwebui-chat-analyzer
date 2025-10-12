from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

import streamlit as st

F = TypeVar("F", bound=Callable[..., Any])


def cache_data(*, ttl: Optional[int] = None, show_spinner: bool = False) -> Callable[[F], F]:
    """Thin wrapper around st.cache_data to centralise caching policy."""

    def decorator(func: F) -> F:
        cached = st.cache_data(ttl=ttl, show_spinner=show_spinner)(func)
        return cached  # type: ignore[return-value]

    return decorator


def cache_resource(*, ttl: Optional[int] = None, show_spinner: bool = False) -> Callable[[F], F]:
    """Thin wrapper around st.cache_resource."""

    def decorator(func: F) -> F:
        cached = st.cache_resource(ttl=ttl, show_spinner=show_spinner)(func)
        return cached  # type: ignore[return-value]

    return decorator

