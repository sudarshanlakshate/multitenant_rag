
from dotenv import load_dotenv

load_dotenv()

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Tenant RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# EXPLICIT NAVIGATION
# ============================================================
#
# IMPORTANT:
# Using st.navigation() means Streamlit will NOT automatically
# expose app.py as a page.
#
# Only these two pages will appear:
#   🔐 Login
#   🤖 RAG
#
# The pages/ directory is ignored for automatic discovery once
# st.navigation() is used.
# ============================================================

pages = [
    st.Page(
        "pages/login.py",
        title="Login",
        icon="🔐",
        url_path="login",
    ),
    st.Page(
        "pages/rag.py",
        title="RAG",
        icon="🤖",
        url_path="rag",
    ),
]


# ============================================================
# RUN CURRENT PAGE
# ============================================================

pg = st.navigation(
    pages,
    position="sidebar",
)

pg.run()

