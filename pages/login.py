
import streamlit as st


# ============================================================
# ALREADY AUTHENTICATED
# ============================================================

if st.user.is_logged_in:

    st.title("🔐 Login")

    st.success(
        f"Logged in as {st.user.name}"
    )

    st.write(
        "You are already authenticated."
    )

    if st.button(
        "🤖 Go to RAG",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/rag.py")

    if st.button(
        "🚪 Logout",
        use_container_width=True,
    ):
        st.logout()

    st.stop()


# ============================================================
# LOGIN
# ============================================================

st.title("🔐 Login")

st.write(
    "Sign in with your Google account to access "
    "the multi-tenant RAG application."
)

st.divider()


if st.button(
    "🔵 Login with Google",
    type="primary",
    use_container_width=True,
):
    st.login("google")

