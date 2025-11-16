import base64
import os
import functools

import boto3
import streamlit as st

from src.config.env_loader import SETTINGS
from src.ui.common import logo_path, APP_NAME

APP_USERNAME = os.getenv("PPE_APP_USERNAME", "admin")  # can parameterize later


@functools.lru_cache(maxsize=1)
def _fetch_password_from_ssm() -> str:
    """
    Fetch the admin password from SSM Parameter Store.

    Looks for SSM param as a SecureString or String.
    Assumes the code is running on EC2 with an IAM role that
    can call ssm:GetParameter on that name.
    """
    ssm = boto3.client("ssm", region_name=SETTINGS.AWS_REGION)
    response = ssm.get_parameter(
        Name="ppe_admin_password",
        WithDecryption=True
    )
    return response["Parameter"]["Value"]


def get_admin_password() -> str:
    """
    Resolve the admin password from SSM Parameter Store via IAM role.
    """
    return _fetch_password_from_ssm()


def require_login():
    """
    Gate the PPE app behind a simple username/password login.

    - Stores 'authenticated' in st.session_state.
    - If not authenticated, renders a login form and stops the app.
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    # ----- simple vertical centering (push card down) -----
    st.markdown(
        "<div style='height: 12vh'></div>",
        unsafe_allow_html=True,
    )

    # --- Center column layout ---
    _, center, _ = st.columns([1.5, 1, 1.5])

    with center:
        # small inner padding
        st.markdown(
            "<div style='padding: 1.25rem 1.5rem;'>",
            unsafe_allow_html=True,
        )

        # --- Logo + title ---
        path = logo_path()
        app_heading = f"{APP_NAME} - Login"
        if path:
            svg_text = path.read_text(encoding="utf-8")
            b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
            st.markdown(
                f"""
                <div style="
                    background: #f3f4ff;
                    padding: 0.75rem 1rem;
                    border-radius: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.6rem;
                    margin-bottom: 1.25rem;
                ">
                    <img src="data:image/svg+xml;base64,{b64}" width="36" />
                    <span style="font-weight: 600; font-size: 1rem;">
                        {app_heading}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: #f3f4ff;
                    padding: 0.75rem 1rem;
                    border-radius: 0.75rem;
                    margin-bottom: 1.25rem;
                ">
                    <span style="font-weight: 600; font-size: 1rem;">
                        {app_heading}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.container(border=True):
            # --- Username / Password fields ---
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            # space before button
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

            # --- Login button ---
            if st.button("Login"):
                try:
                    expected_password = get_admin_password()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unable to retrieve application credentials. Please contact the administrator. ({exc})")
                    st.stop()

                if username == APP_USERNAME and password == expected_password:
                    st.session_state["authenticated"] = True
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    # User not authenticated yet; stop the app so pages don't render
    st.stop()
