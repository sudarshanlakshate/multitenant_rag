from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings

from src.db.models import Chunk, Document, DocumentVersion
from src.db.session import SessionLocal

from src.ingestion.chunking.splitter import StructuredChunk, chunk_text
from src.ingestion.loaders.html import parse_html
from src.ingestion.loaders.office import parse_office
from src.ingestion.loaders.pdf import parse_pdf
from src.ingestion.loaders.text import parse_text


logger = logging.getLogger(__name__)


# ============================================================
# IN-PROCESS EMBEDDING CACHE
# ============================================================
#
# Keyed by (tenant_id, content_hash) → document_id.
#
# This is a process-level guard: if the same file is uploaded
# twice in the same server process (e.g. Streamlit rerun),
# we never even open a DB connection for the duplicate check.
#
# The DB-level check in _persist_document_metadata is the
# authoritative guard across process restarts.
# ============================================================

_INGESTED_CACHE: dict[tuple[str, str], str] = {}


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_DIR = Path("processed_data")

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


PARSER_REGISTRY: dict[str, Callable[[str], str]] = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
    ".md": parse_text,
    ".docx": parse_office,
    ".pptx": parse_office,
}


# ============================================================
# FILE HELPERS
# ============================================================

def _safe_filename(filename: str) -> str:
    """
    Prevent path traversal and unsafe filenames in local
    processed-data output.

    Example:

        ../../secret.pdf

    becomes:

        secret.pdf
    """

    name = Path(filename).name

    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )


def calculate_sha256(file_path: str) -> str:
    """
    Calculate SHA-256 without loading the entire file into RAM.

    Files are processed in 1 MB blocks.
    """

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def calculate_chunk_hash(text: str) -> str:
    """
    Calculate a stable SHA-256 hash for an individual chunk.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_file(file_path: str) -> Path:
    """
    Validate basic file properties before invoking a parser.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    size = path.stat().st_size

    if size == 0:
        raise ValueError(
            f"File is empty: {file_path}"
        )

    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds "
            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB: "
            f"{file_path}"
        )

    if path.suffix.lower() not in PARSER_REGISTRY:
        supported = ", ".join(
            sorted(PARSER_REGISTRY)
        )

        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {supported}"
        )

    return path


# ============================================================
# DOCUMENT IDENTIFIERS
# ============================================================

def build_document_id(
    tenant_id: str,
    filename: str,
) -> str:
    """
    Build a stable logical document identity.

    The identity is based on:

        tenant_id + filename

    Therefore:

        tenant A + report.pdf
            !=
        tenant B + report.pdf

    But:

        tenant A + report.pdf
            ==
        tenant A + report.pdf

    across multiple uploads.

    Content changes are tracked using DocumentVersion.
    """

    key = (
        f"{tenant_id}:"
        f"{Path(filename).name.lower()}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            key,
        )
    )


def build_point_id(
    tenant_id: str,
    document_id: str,
    chunk_index: int,
) -> str:
    """
    Build a stable Chroma vector ID.

    The ID is deterministic for:

        tenant
        document
        chunk position
    """

    key = (
        f"{tenant_id}:"
        f"{document_id}:"
        f"{chunk_index}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            key,
        )
    )


# ============================================================
# LOCAL AUDIT ARTIFACT
# ============================================================

def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:
    """
    Persist a local audit/debug artifact.

    This is NOT the system of record.

    PostgreSQL is the source of truth for document lineage.
    """

    folder = (
        PROCESSED_DATA_DIR
        / source_type
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = _safe_filename(
        filename
    )

    dest = folder / f"{safe_name}.json"

    with dest.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return str(dest)


# ============================================================
# CACHED EMBEDDINGS WRAPPER
# ============================================================

class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Enterprise-grade in-memory embedding cache:
    1. embed_query() caches query vectors to eliminate duplicate OpenAI API calls
       for identical or decomposed queries.
    2. embed_documents() caches chunk vectors by content SHA-256 hash.
    """
    _query_cache: dict[str, list[float]] = {}
    _chunk_cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        key = text.strip()
        if key in self._query_cache:
            logger.debug("[EMBED CACHE HIT] Query: '%s'", key[:50])
            return self._query_cache[key]
        vector = super().embed_query(text)
        self._query_cache[key] = vector
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in self._chunk_cache:
                results[idx] = self._chunk_cache[h]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            fresh_vectors = super().embed_documents(missing_texts)
            for idx, text, vec in zip(missing_indices, missing_texts, fresh_vectors):
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self._chunk_cache[h] = vec
                results[idx] = vec

        return [r for r in results if r is not None]


# ============================================================
# CHROMA
# ============================================================

def _get_chroma_collection(
    tenant_id: str,
) -> Chroma:
    """
    Return a tenant-scoped ChromaDB collection.

    Every tenant receives its own collection:

        tenant_<tenant_id>

    This prevents vectors from different tenants
    from being placed into the same collection.
    """

    embeddings = CachedOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv(
            "OPENAI_API_KEY"
        ),
    )

    clean_tenant = re.sub(
        r"[^a-z0-9_-]",
        "_",
        str(tenant_id).lower(),
    )

    collection_name = (
        f"tenant_{clean_tenant}"
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.getenv("CHROMA_PATH", "./chroma_db"),
    )


def delete_existing_document(
    chroma: Chroma,
    document_id: str,
) -> None:
    """
    Remove old Chroma chunks for the same logical document.

    This keeps the active Chroma representation aligned with
    the newest successfully indexed version.
    """

    try:
        results = chroma.get(
            where={
                "document_id": document_id
            }
        )

        ids_to_delete = results.get(
            "ids",
            [],
        )

        if ids_to_delete:
            chroma.delete(
                ids=ids_to_delete
            )

            logger.info(
                "Deleted %d existing Chroma chunks "
                "for document_id=%s",
                len(ids_to_delete),
                document_id,
            )

    except Exception as exc:
        logger.warning(
            "Could not delete existing Chroma chunks "
            "for document_id=%s: %s",
            document_id,
            exc,
        )


# ============================================================
# POSTGRESQL PERSISTENCE
# ============================================================

def _persist_document_metadata(
    db,
    *,
    tenant_id: str,
    document_id: str,
    filename: str,
    content_hash: str,
    chunks: list[StructuredChunk],
    source_type: str,
    file_type: str,
    user_id: str | None,
) -> tuple[Document, DocumentVersion]:
    """
    Persist document/version/chunk lineage into PostgreSQL.

    PostgreSQL hierarchy:

        documents
            |
            +-- document_versions
                    |
                    +-- chunks

    This function only prepares the transaction.

    The caller commits after Chroma indexing succeeds.
    """

    tenant_uuid = uuid.UUID(
        tenant_id
    )

    document_uuid = uuid.UUID(
        document_id
    )

    # --------------------------------------------------------
    # 1. Find existing logical document
    # --------------------------------------------------------

    document = db.scalar(
        select(Document).where(
            Document.id == document_uuid,
            Document.tenant_id == tenant_uuid,
        )
    )

    # --------------------------------------------------------
    # 2. Create logical document if this is the first upload
    # --------------------------------------------------------

    if document is None:

        document = Document(
            id=document_uuid,
            tenant_id=tenant_uuid,
            source_uri=filename,
            content_hash=content_hash,
            latest_version=0,
            active_version=None,
            status="INDEXING",
        )

        db.add(document)

        # Make sure SQLAlchemy assigns the object
        # to the current transaction before continuing.
        db.flush()

        logger.info(
            "Created PostgreSQL document: "
            "document_id=%s tenant_id=%s",
            document_id,
            tenant_id,
        )

    # --------------------------------------------------------
    # 3. Idempotency check
    # --------------------------------------------------------
    #
    # If this exact content is already the active version,
    # there is nothing new to persist.
    # --------------------------------------------------------

    if (
        document.content_hash == content_hash
        and document.active_version is not None
    ):
        existing_version = db.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id
                == document_uuid,
                DocumentVersion.version
                == document.active_version,
            )
        )

        if existing_version is not None:
            logger.info(
                "Document already indexed with identical "
                "content: document_id=%s version=%s",
                document_id,
                existing_version.version,
            )

            return (
                document,
                existing_version,
            )

    # --------------------------------------------------------
    # 4. Determine next version
    # --------------------------------------------------------

    next_version = (
        int(document.latest_version)
        + 1
    )

    # --------------------------------------------------------
    # 5. Create document version
    # --------------------------------------------------------

    document_version = DocumentVersion(
        document_id=document_uuid,
        version=next_version,
        content_hash=content_hash,
        status="PENDING",
        chunk_count=len(chunks),
    )

    db.add(
        document_version
    )

    # --------------------------------------------------------
    # 6. Create PostgreSQL chunk records
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):

        chunk_hash = calculate_chunk_hash(
            chunk.text
        )

        chunk = Chunk(
            tenant_id=tenant_uuid,
            document_id=document_uuid,
            version=next_version,
            chunk_index=index,
            text=chunk.text,
            chunk_hash=chunk_hash,
            section_path=chunk.section_path,
            page=chunk.page,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            document_metadata={
                "filename": filename,
                "source_type": source_type,
                "file_type": file_type,
                "content_hash": content_hash,
                "user_id": user_id,
            },
        )

        db.add(chunk)

    # --------------------------------------------------------
    # 7. Update logical document
    # --------------------------------------------------------

    document.latest_version = (
        next_version
    )

    document.content_hash = (
        content_hash
    )

    document.source_uri = filename

    document.status = "INDEXING"

    # Ensure all INSERT/UPDATE statements are sent to
    # PostgreSQL before Chroma indexing begins.
    db.flush()

    logger.info(
        "Prepared PostgreSQL document version: "
        "document_id=%s version=%s chunks=%d",
        document_id,
        next_version,
        len(chunks),
    )

    return (
        document,
        document_version,
    )


def _activate_document_version(
    document: Document,
    document_version: DocumentVersion,
) -> None:
    """
    Mark a successfully indexed document version as ACTIVE.

    This is called only after Chroma ingestion succeeds.
    """

    document_version.status = "ACTIVE"

    document_version.activated_at = (
        datetime.now(timezone.utc)
    )

    document.active_version = (
        document_version.version
    )

    document.status = "ACTIVE"


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def process_file(
    file_path: str,
    filename: str | None = None,
    source_type: str = "general",
    *,
    tenant_id: str,
    user_id: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """
    Production ingestion flow:

        validate
            ↓
        hash
            ↓
        parse
            ↓
        chunk
            ↓
        persist PostgreSQL metadata/chunks
            ↓
        embed
            ↓
        index Chroma
            ↓
        activate PostgreSQL version
            ↓
        commit transaction

    PostgreSQL stores document lineage.

    Chroma stores vector embeddings.

    Local JSON is only an audit/debug artifact.
    """

    tenant_id = str(
        tenant_id
    )

    user_id = (
        str(user_id)
        if user_id is not None
        else None
    )

    path = validate_file(
        file_path
    )

    filename = (
        filename
        or path.name
    )

    logger.info(
        "Processing file: "
        "filename=%s source=%s tenant_id=%s",
        filename,
        source_type,
        tenant_id,
    )

    db = SessionLocal()

    try:

        # ====================================================
        # 1. CALCULATE FILE HASH
        # ====================================================

        content_hash = calculate_sha256(
            str(path)
        )

        # ====================================================
        # 2. BUILD STABLE DOCUMENT ID
        # ====================================================

        document_id = build_document_id(
            tenant_id,
            filename,
        )

        parser = PARSER_REGISTRY[
            path.suffix.lower()
        ]

        logger.info(
            "Document parsing started: "
            "filename=%s parser=%s document_id=%s",
            filename,
            parser.__name__,
            document_id,
        )

        # ====================================================
        # 3. PARSE DOCUMENT
        # ====================================================

        full_text = parser(
            str(path)
        )

        if not full_text.strip():

            raise ValueError(
                f"No usable text extracted from "
                f"'{filename}'. "
                "The document may be scanned/image-only "
                "or malformed."
            )

        # ====================================================
        # 4. CHUNK DOCUMENT
        # ====================================================

        chunks = chunk_text(
            full_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        if not chunks:

            raise ValueError(
                f"No chunks generated "
                f"for '{filename}'."
            )

        # ====================================================
        # 5. LOCAL AUDIT ARTIFACT
        # ====================================================

        processed_data = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "filename": filename,
            "source_type": source_type,
            "file_type": path.suffix.lower(),
            "content_hash": content_hash,
            "character_count": len(
                full_text
            ),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_index": index,
                    "text": chunk.text,
                    "page": chunk.page,
                    "section_path": chunk.section_path,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
                for index, chunk in enumerate(
                    chunks
                )
            ],
        }

        local_path = save_processed_locally(
            processed_data,
            source_type,
            f"{document_id}_{filename}",
        )

        # ====================================================
        # 6. LAYER 1 CACHE — in-process RAM check
        # ====================================================
        #
        # If the same tenant uploaded the same content bytes
        # in this server process, skip everything immediately.
        # ====================================================

        cache_key = (tenant_id, content_hash)

        if cache_key in _INGESTED_CACHE:
            cached_doc_id = _INGESTED_CACHE[cache_key]
            logger.info(
                "[CACHE HIT] Document already embedded this session: "
                "tenant_id=%s document_id=%s content_hash=%s",
                tenant_id, cached_doc_id, content_hash,
            )
            db.close()
            return {
                "status": "skipped",
                "reason": "in_process_cache_hit",
                "document_id": cached_doc_id,
                "tenant_id": tenant_id,
                "filename": filename,
                "content_hash": content_hash,
                "version": None,
                "chunks": 0,
                "characters": 0,
                "processed_path": local_path,
            }

        # ====================================================
        # 7. PERSIST DOCUMENT + VERSION + CHUNKS (DB layer)
        # ====================================================

        document, document_version = (
            _persist_document_metadata(
                db,
                tenant_id=tenant_id,
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                chunks=chunks,
                source_type=source_type,
                file_type=path.suffix.lower(),
                user_id=user_id,
            )
        )

        # ====================================================
        # 8. LAYER 2 CACHE — DB version already ACTIVE?
        # ====================================================
        #
        # _persist_document_metadata returns the existing
        # DocumentVersion when content_hash hasn't changed.
        # That version already has status=ACTIVE, meaning
        # Chroma should already have these vectors.
        #
        # IMPORTANT: the DB is authoritative for lineage, but
        # NOT for Chroma state. Chroma can be wiped externally
        # (reset_collection, manual deletion, storage failure).
        # We therefore verify Chroma actually has the vectors
        # before skipping; if they are missing, we fall through
        # to the normal embed path so retrieval never breaks.
        # ====================================================

        if document_version.status == "ACTIVE":
            try:
                _probe_chroma = _get_chroma_collection(tenant_id)
                _probe_ids = [
                    build_point_id(tenant_id, document_id, i)
                    for i in range(len(chunks))
                ]
                _probe_result = _probe_chroma.get(ids=_probe_ids)
                _probe_existing = set(_probe_result.get("ids", []))
            except Exception:
                _probe_existing = set()

            if len(_probe_existing) == len(chunks):
                logger.info(
                    "[DB CACHE HIT] Document unchanged and all %d Chroma "
                    "vectors present — skipping re-embedding: "
                    "tenant_id=%s document_id=%s version=%s",
                    len(chunks), tenant_id, document_id, document_version.version,
                )
                # Populate RAM cache so the next call in this process is instant.
                _INGESTED_CACHE[cache_key] = document_id
                db.close()
                return {
                    "status": "skipped",
                    "reason": "already_indexed",
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "filename": filename,
                    "content_hash": content_hash,
                    "version": document_version.version,
                    "chunks": document_version.chunk_count,
                    "characters": len(full_text),
                    "processed_path": local_path,
                }

            logger.warning(
                "[DB CACHE MISS] DB says ACTIVE but Chroma is missing %d/%d "
                "vectors — re-embedding: tenant_id=%s document_id=%s",
                len(chunks) - len(_probe_existing), len(chunks),
                tenant_id, document_id,
            )

        # ====================================================
        # 9. BUILD LANGCHAIN DOCUMENTS
        # ====================================================

        lc_documents: list[LCDocument] = []

        ids: list[str] = []

        for index, chunk in enumerate(
            chunks
        ):

            point_id = build_point_id(
                tenant_id,
                document_id,
                index,
            )

            lc_documents.append(
                LCDocument(
                    page_content=chunk.text,
                    metadata={
                        "tenant_id": tenant_id,
                        "user_id": user_id or "",
                        "document_id": document_id,
                        "version": document_version.version,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                        "source": filename,
                        "source_type": source_type,
                        "file_type": path.suffix.lower(),
                        "content_hash": content_hash,
                        # Stable per-chunk point ID. RRF merges by this key,
                        # so dense and BM25 must share the same ID namespace.
                        "chunk_id": point_id,
                        "page": chunk.page,
                        "section_path": chunk.section_path,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                )
            )

            ids.append(
                point_id
            )

        # ====================================================
        # 10. GET TENANT-SCOPED CHROMA COLLECTION
        # ====================================================

        chroma = _get_chroma_collection(
            tenant_id
        )

        # ====================================================
        # 11. LAYER 3 CACHE — check Chroma before API call
        # ====================================================
        #
        # Point IDs are deterministic (uuid5 of tenant+doc+index).
        # If all IDs already exist in Chroma, the vectors are
        # already indexed — skip the OpenAI embedding API call.
        # ====================================================

        try:
            existing_result = chroma.get(ids=ids)
            existing_ids: set[str] = set(existing_result.get("ids", []))
        except Exception:
            existing_ids = set()

        ids_to_embed = [i for i in ids if i not in existing_ids]
        docs_to_embed = [
            doc for doc, pid in zip(lc_documents, ids)
            if pid not in existing_ids
        ]

        if not ids_to_embed:
            logger.info(
                "[CHROMA CACHE HIT] All %d chunks already in Chroma — "
                "no OpenAI embedding API call needed: "
                "tenant_id=%s document_id=%s",
                len(ids), tenant_id, document_id,
            )
        else:
            logger.info(
                "Embedding %d new / %d already-cached chunks: "
                "tenant_id=%s document_id=%s",
                len(ids_to_embed), len(existing_ids),
                tenant_id, document_id,
            )

            # ================================================
            # 12. REMOVE OLD VERSION VECTORS (only if needed)
            # ================================================
            #
            # Only delete if we are actually writing new vectors.
            # This keeps Chroma consistent with the newest version.
            # ================================================

            delete_existing_document(
                chroma,
                document_id,
            )

            # ================================================
            # 13. EMBED + WRITE TO CHROMA
            # ================================================

            chroma.add_documents(
                documents=docs_to_embed,
                ids=ids_to_embed,
            )

        logger.info(
            "Chroma indexing completed: "
            "tenant_id=%s document_id=%s "
            "version=%s new_chunks=%d cached_chunks=%d",
            tenant_id,
            document_id,
            document_version.version,
            len(ids_to_embed),
            len(existing_ids),
        )

        # Populate RAM cache so the next upload in this process is instant.
        _INGESTED_CACHE[cache_key] = document_id

        # ====================================================
        # 11. ACTIVATE VERSION
        # ====================================================

        _activate_document_version(
            document,
            document_version,
        )

        # ====================================================
        # 12. COMMIT POSTGRESQL TRANSACTION
        # ====================================================

        db.commit()

        # ====================================================
        # 13. RETURN RESULT
        # ====================================================

        result = {
            "status": "completed",
            "document_id": document_id,
            "tenant_id": tenant_id,
            "filename": filename,
            "version": document_version.version,
            "chunks": len(chunks),
            "characters": len(full_text),
            "content_hash": content_hash,
            "processed_path": local_path,
        }

        logger.info(
            "Document ingestion completed: %s",
            result,
        )

        return result

    except Exception:

        # ----------------------------------------------------
        # PostgreSQL rollback
        # ----------------------------------------------------

        db.rollback()

        logger.exception(
            "Document ingestion failed: "
            "filename=%s tenant_id=%s",
            filename,
            tenant_id,
        )

        raise

    finally:

        # ----------------------------------------------------
        # Always release database connection.
        # ----------------------------------------------------

        db.close()


# ============================================================
# ASYNC / BACKGROUND INGESTION
# ============================================================

import concurrent.futures

_INGESTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="rag_ingest",
)


def process_file_async(
    file_path: str,
    filename: str,
    source_type: str = "document",
    *,
    tenant_id: str,
    user_id: str | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> concurrent.futures.Future[dict]:
    """
    Submit document processing to a background worker thread.
    Returns a Future that resolves with the ingestion result dictionary.
    """
    return _INGESTION_EXECUTOR.submit(
        process_file,
        file_path,
        filename,
        source_type,
        tenant_id=tenant_id,
        user_id=user_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )


# ============================================================
# DIRECTORY INGESTION
# ============================================================

def process_directory(
    dir_path: str,
    source_type: str,
    *,
    tenant_id: str,
    user_id: str | None = None,
) -> list[dict]:
    """
    Process all supported files in one directory.
    """

    directory = Path(
        dir_path
    )

    if not directory.is_dir():

        raise NotADirectoryError(
            f"Directory not found: {dir_path}"
        )

    results: list[dict] = []

    for path in sorted(
        directory.iterdir()
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in PARSER_REGISTRY
        ):

            logger.warning(
                "Skipping unsupported file: "
                "filename=%s extension=%s",
                path.name,
                path.suffix,
            )

            continue

        try:

            results.append(
                process_file(
                    str(path),
                    path.name,
                    source_type,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
            )

        except Exception as exc:

            # Continue processing other documents
            # in the same batch.

            results.append(
                {
                    "status": "failed",
                    "filename": path.name,
                    "tenant_id": tenant_id,
                    "error": str(exc),
                }
            )

    return results


# ============================================================
# UNIVERSAL INGESTION
# ============================================================

def run_universal_ingestion(
    base_dir: str,
    *,
    tenant_id: str,
    user_id: str | None = None,
    explicit_source_type: str | None = None,
    wipe: bool = False,
) -> list[dict]:
    """
    Ingest a directory tree.

    wipe=True is intended for development/testing only.

    It resets the tenant's Chroma collection.

    PostgreSQL is NOT wiped by this function.
    """

    logger.info(
        "Universal Ingestion Started: "
        "base_directory=%s tenant_id=%s",
        base_dir,
        tenant_id,
    )

    # ========================================================
    # OPTIONAL CHROMA WIPE
    # ========================================================

    if wipe:

        logger.warning(
            "WIPE requested: resetting ChromaDB "
            "collection for tenant=%s",
            tenant_id,
        )

        chroma = _get_chroma_collection(
            tenant_id
        )

        chroma.reset_collection()

    # ========================================================
    # VALIDATE BASE DIRECTORY
    # ========================================================

    base_path = Path(
        base_dir
    )

    if not base_path.is_dir():

        raise NotADirectoryError(
            f"Path not found: {base_dir}"
        )

    subdirs = sorted(
        path
        for path in base_path.iterdir()
        if path.is_dir()
    )

    # ========================================================
    # NO SUBDIRECTORIES
    # ========================================================

    if not subdirs:

        source_type = (
            explicit_source_type
            or "general"
        )

        return process_directory(
            str(base_path),
            source_type,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    # ========================================================
    # PROCESS SUBDIRECTORIES
    # ========================================================

    results: list[dict] = []

    for subdir in subdirs:

        source_type = (
            "true"
            if "true"
            in subdir.name.lower()
            else "noisy"
            if "noisy"
            in subdir.name.lower()
            else subdir.name
        )

        results.extend(
            process_directory(
                str(subdir),
                source_type,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        )

    return results


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run multi-tenant "
            "document ingestion."
        )
    )

    parser.add_argument(
        "data_dir",
        nargs="?",
        default="DATA",
    )

    parser.add_argument(
        "--tenant-id",
        required=True,
    )

    parser.add_argument(
        "--user-id",
        default=None,
    )

    parser.add_argument(
        "--source-type",
        default=None,
    )

    parser.add_argument(
        "--wipe",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = _parse_args()

    summary = run_universal_ingestion(
        args.data_dir,
        tenant_id=args.tenant_id,
        user_id=args.user_id,
        explicit_source_type=args.source_type,
        wipe=args.wipe,
    )

    completed = sum(
        item["status"] == "completed"
        for item in summary
    )

    failed = sum(
        item["status"] == "failed"
        for item in summary
    )

    print(
        f"Ingestion complete: "
        f"{completed} succeeded, "
        f"{failed} failed."
    )

    if failed:
        raise SystemExit(1)