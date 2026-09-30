from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models import Tenant, User


def get_or_create_user(
    db: Session,
    *,
    google_subject: str,
    email: str,
    name: str | None,
) -> User:
    """
    Find an existing application user by Google's stable `sub`.

    If the user does not exist:

        1. Create a tenant.
        2. Create the application user.
        3. Link the user to that tenant.
        4. Commit everything in one transaction.

    This function is intentionally idempotent.

    Repeated calls for the same Google account return
    the same application user and tenant.
    """

    google_subject = google_subject.strip()
    email = email.strip()

    if not google_subject:
        raise ValueError("google_subject must not be empty.")

    if not email:
        raise ValueError("email must not be empty.")

    # ---------------------------------------------------------
    # 1. Check whether this Google user already exists.
    # ---------------------------------------------------------

    statement = select(User).where(
        User.google_subject == google_subject
    )

    existing_user: User | None = db.scalar(statement)

    if existing_user is not None:

        # Keep profile information reasonably synchronized
        # with the latest Google identity information.
        existing_user.email = email
        existing_user.name = name

        db.commit()
        db.refresh(existing_user)

        return existing_user

    # ---------------------------------------------------------
    # 2. First login -> create a new tenant.
    # ---------------------------------------------------------

    tenant_id: uuid.UUID = uuid.uuid4()

    tenant = Tenant(
        id=tenant_id,
        name=f"{name or email}'s Workspace",
        slug=f"tenant-{tenant_id.hex}",
        status="ACTIVE",
    )

    db.add(tenant)

    # ---------------------------------------------------------
    # 3. Create the application user.
    # ---------------------------------------------------------

    user = User(
        tenant_id=tenant_id,
        google_subject=google_subject,
        email=email,
        name=name,
        role="OWNER",
        status="ACTIVE",
    )

    db.add(user)

    # ---------------------------------------------------------
    # 4. Persist both records atomically.
    # ---------------------------------------------------------

    try:
        db.commit()

    except IntegrityError:
        # Another request may have created this same Google
        # user between our SELECT and INSERT.
        #
        # Roll back the failed transaction before querying again.
        db.rollback()

        existing_user = db.scalar(
            select(User).where(
                User.google_subject == google_subject
            )
        )

        if existing_user is None:
            raise

        return existing_user

    db.refresh(user)

    return user