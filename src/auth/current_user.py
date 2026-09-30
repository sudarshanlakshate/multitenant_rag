from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from src.auth.user_service import get_or_create_user
from src.db.models import User


def provision_authenticated_user(db: Session) -> User:
    """
    Synchronize the currently authenticated Google user
    with our PostgreSQL application database.

    Streamlit authentication remains responsible for
    authentication.

    PostgreSQL is responsible for application identity,
    tenant ownership, roles, and persistence.
    """

    if not st.user.is_logged_in:
        raise RuntimeError(
            "Cannot provision a user who is not authenticated."
        )

    google_subject = getattr(st.user, "sub", None)
    email = getattr(st.user, "email", None)
    name = getattr(st.user, "name", None)

    if not isinstance(google_subject, str):
        raise RuntimeError(
            "Authenticated Google user is missing the `sub` claim."
        )

    if not isinstance(email, str):
        raise RuntimeError(
            "Authenticated Google user is missing an email."
        )

    normalized_name: str | None

    if isinstance(name, str):
        normalized_name = name
    else:
        normalized_name = None

    return get_or_create_user(
        db,
        google_subject=google_subject,
        email=email,
        name=normalized_name,
    )