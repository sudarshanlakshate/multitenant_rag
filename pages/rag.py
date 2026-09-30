
from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

import importlib

from src.auth.current_user import provision_authenticated_user
from src.db.models import Tenant
from src.db.session import SessionLocal

# Production ingestion & retrieval pipeline
import src.ingestion.processor
import src.query.rag

importlib.reload(src.ingestion.processor)
importlib.reload(src.query.rag)

from src.ingestion.processor import process_file
from src.query.rag import query_tenant_rag


# ============================================================
# AUTHENTICATION GUARD
# ============================================================

if not st.user.is_logged_in:

    st.error("🔒 Authentication required.")

    st.write(
        "Please login with Google before accessing the RAG application."
    )

    if st.button(
        "🔐 Go to Login",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/login.py")

    st.stop()


# ============================================================
# DATABASE USER / TENANT PROVISIONING
# ============================================================

db = SessionLocal()

try:
    current_user = provision_authenticated_user(db)
    tenant = db.get(Tenant, current_user.tenant_id)
    tenant_name = tenant.name if tenant else str(current_user.tenant_id)

finally:
    db.close()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Multi-Tenant RAG")

st.write(
    f"Welcome, **{current_user.name or current_user.email}**"
)

st.caption(
    f"Logged in as: {current_user.email}"
)

st.caption(
    f"Tenant Name: **{tenant_name}**"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👤 Session")

    st.write(
        current_user.name or current_user.email
    )

    st.caption(
        f"Tenant: {tenant_name}"
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True,
    ):
        st.logout()


st.divider()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("📄 Upload Documents")

st.write(
    "Upload a document to parse, chunk, embed, and index it "
    "inside your tenant's isolated ChromaDB namespace."
)


uploaded_files = st.file_uploader(
    "Choose documents",
    type=[
        "pdf",
        "docx",
        "pptx",
        "html",
        "htm",
        "txt",
        "md",
    ],
    accept_multiple_files=True,
    max_upload_size=50,
    help=(
        "Supported: PDF, DOCX, PPTX, HTML, TXT and Markdown. "
        "Maximum 50 MB per file."
    ),
)

with st.expander("⚙️ Ingestion & Chunking Precision Settings", expanded=False):
    col_cs, col_co = st.columns(2)
    with col_cs:
        chunk_size_setting = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2000,
            value=500,
            step=50,
            help="Smaller chunks (400-600) provide high-precision answers without bloating context. Larger chunks (1000+) are suitable for broad overviews.",
        )
    with col_co:
        chunk_overlap_setting = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Overlap prevents boundary fragmentation between adjacent paragraphs.",
        )


# ============================================================
# INGEST DOCUMENTS
# ============================================================

if uploaded_files:

    st.write(
        f"**{len(uploaded_files)} file(s) selected.**"
    )

    for uploaded_file in uploaded_files:

        st.divider()

        filename = uploaded_file.name
        suffix = Path(filename).suffix.lower()

        st.write(
            f"📄 **{filename}**"
        )

        st.caption(
            f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            f"🚀 Process {filename}",
            key=f"process_{filename}",
            type="primary",
        ):

            temp_path = None

            try:

                # ------------------------------------------------
                # 1. Read uploaded file
                # ------------------------------------------------

                file_bytes = uploaded_file.getvalue()

                if not file_bytes:

                    st.error(
                        f"{filename} is empty."
                    )

                    continue

                # ------------------------------------------------
                # 2. Create temporary server-side file
                # ------------------------------------------------
                #
                # Your ingestion pipeline expects a file path.
                #
                # We therefore bridge:
                #
                # Streamlit UploadedFile
                #          ↓
                # Temporary file
                #          ↓
                # process_file()
                #
                # The original uploaded filename is preserved
                # separately.
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:

                    temp_file.write(file_bytes)

                    temp_path = temp_file.name

                # ------------------------------------------------
                # 3. Run production ingestion pipeline
                # ------------------------------------------------

                with st.status(
                    f"Processing `{filename}`...",
                    expanded=True,
                ) as status:

                    st.write("🔍 Validating file...")

                    st.write("📖 Parsing document...")

                    st.write("✂️ Creating chunks...")

                    st.write("🧠 Generating embeddings...")

                    st.write("🗄️ Indexing into ChromaDB...")

                    result = process_file(
                        file_path=temp_path,
                        filename=filename,
                        source_type="streamlit_upload",
                        tenant_id=str(current_user.tenant_id),
                        user_id=str(current_user.id) if current_user.id else None,
                        chunk_size=chunk_size_setting,
                        chunk_overlap=chunk_overlap_setting,
                    )

                    status.update(
                        label=f"✅ {filename} processed successfully",
                        state="complete",
                        expanded=False,
                    )

                # ------------------------------------------------
                # 4. Show ingestion result
                # ------------------------------------------------

                if result.get("status") == "skipped":
                    # ------------------------------------------------
                    # Document already indexed — no re-embedding done
                    # ------------------------------------------------
                    st.info(
                        f"✅ `{filename}` is **already indexed** "
                        f"(version {result.get('version', '—')}, "
                        f"hash `{result['content_hash'][:12]}…`). "
                        f"No re-processing or API calls were made."
                    )

                else:
                    st.success(
                        f"Successfully indexed `{filename}`."
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Chunks",
                            result["chunks"],
                        )

                    with col2:

                        st.metric(
                            "Characters",
                            result["characters"],
                        )

                    with col3:

                        st.metric(
                            "Status",
                            "Indexed",
                        )

                with st.expander(
                    "Document details"
                ):

                    st.write(
                        {
                            "document_id": result["document_id"],
                            "tenant_id": result["tenant_id"],
                            "filename": result["filename"],
                            "content_hash": result["content_hash"],
                            "processed_path": result.get("processed_path", "—"),
                            "status": result.get("status"),
                            "reason": result.get("reason", "new_document"),
                        }
                    )

            except Exception as exc:

                st.error(
                    f"❌ Failed to process `{filename}`"
                )

                st.exception(exc)

            finally:

                # ------------------------------------------------
                # 5. Delete temporary upload
                # ------------------------------------------------

                if temp_path:

                    try:

                        Path(temp_path).unlink(
                            missing_ok=True
                        )

                    except Exception:

                        # Do not hide a successful ingestion
                        # just because temporary cleanup failed.
                        pass


# ============================================================
# RAG QUERY
# ============================================================

st.divider()

st.header("💬 Ask Your Documents")

with st.form("rag_query_form", clear_on_submit=False):
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. what is company name ?",
    )
    col_k, col_harness = st.columns([1, 1])
    with col_k:
        top_k = st.slider("Retrieval Depth (k chunks)", min_value=1, max_value=6, value=3, help="Number of chunks to retrieve for context synthesis.")
    with col_harness:
        enable_harness = st.checkbox("🧪 Run Harness Evaluation & Telemetry", value=True, help="Runs the evaluation harness to measure faithfulness, relevance, and latency.")

    submit_button = st.form_submit_button("🔍 Ask", type="primary")

if submit_button:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔍 Searching your workspace documents & generating answer..."):
            result = query_tenant_rag(
                query=query.strip(),
                tenant_id=str(current_user.tenant_id),
                k=top_k,
                include_eval=enable_harness,
            )

        st.markdown("### 💡 Answer")
        st.success(result["answer"])

        # Latency & Telemetry Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Inference Engine", result.get("llm_provider", "LLM"))
        with m2:
            st.metric("Total Latency", f"{result.get('total_time_ms', 0):.0f} ms")
        with m3:
            st.metric("Retrieval Time", f"{result.get('retrieval_time_ms', 0):.0f} ms")
        with m4:
            st.metric("Inference Time", f"{result.get('inference_time_ms', 0):.0f} ms")

        # Harness Evaluation Report
        if "evaluation" in result:
            eval_data = result["evaluation"]
            with st.expander("🧪 Harness Evaluation Report", expanded=True):
                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    st.metric("Overall Quality", f"{eval_data['overall_quality_score'] * 100:.0f}%")
                with e2:
                    st.metric("Faithfulness", f"{eval_data['faithfulness_score'] * 100:.0f}%")
                with e3:
                    st.metric("Answer Relevance", f"{eval_data['answer_relevance_score'] * 100:.0f}%")
                with e4:
                    st.metric("Context Match", f"{eval_data['context_relevance_score'] * 100:.0f}%")

                details = eval_data.get("evaluation_details", {})
                if details.get("faithfulness_explanation"):
                    st.caption(f"**Faithfulness Analysis:** {details['faithfulness_explanation']}")
                if details.get("relevance_explanation"):
                    st.caption(f"**Relevance Analysis:** {details['relevance_explanation']}")

        # Granular Sources
        if result.get("sources"):
            with st.expander(f"📚 Retrieved Sources ({result['chunks_found']} chunks)", expanded=False):
                for idx, src in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"**Chunk {idx}** | Source: `{src['source']}` | "
                        f"Chunk #{src['chunk_index']} | "
                        f"Distance: `{src.get('distance', 'N/A')}` | "
                        f"Size: `{src.get('character_count', len(src['content']))} chars`"
                    )
                    st.info(src["preview"])
                    with st.expander(f"Show complete chunk #{src['chunk_index']} text"):
                        st.code(src["content"], language="text")
                    if idx < len(result["sources"]):
                        st.divider()
