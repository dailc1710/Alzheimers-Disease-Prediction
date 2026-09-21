"""Shared layout and visual styling for the Streamlit application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

_STYLES_PATH = Path(__file__).with_name("styles.css")


def inject_app_styles() -> None:
    """Load the dashboard stylesheet from a standalone, cacheable asset."""

    styles = _STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render the persistent product identity above navigation."""

    st.sidebar.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-name">NeuroScreen</div>
                <div class="brand-caption">Alzheimer research workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_intro(translate: Callable[[str, str], str]) -> None:
    """Introduce the workspace beside the sign-in form."""

    st.markdown(
        f"""
        <div class="auth-intro">
            <div class="auth-eyebrow"><span class="auth-eyebrow-dot"></span>{translate("Research workspace", "Không gian nghiên cứu")}</div>
            <h1>{translate("Clear data. Responsible screening.", "Dữ liệu rõ ràng. Sàng lọc có trách nhiệm.")}</h1>
            <p class="auth-lead">{translate(
                "One place to screen individual cases, process CSV data, and review model evidence.",
                "Một nơi để sàng lọc từng ca, xử lý dữ liệu CSV và xem bằng chứng của mô hình.",
            )}</p>
            <div class="auth-feature-list">
                <div><span>01</span>{translate("Screen a case", "Sàng lọc một ca")}</div>
                <div><span>02</span>{translate("Review CSV quality", "Kiểm tra chất lượng CSV")}</div>
                <div><span>03</span>{translate("Track the model", "Theo dõi mô hình")}</div>
            </div>
            <p class="auth-disclaimer">{translate(
                "For education and research only. Not a medical diagnosis or a substitute for a clinician.",
                "Chỉ phục vụ học tập và nghiên cứu; không phải chẩn đoán hoặc thay thế bác sĩ.",
            )}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_heading(
    translate: Callable[[str, str], str], *, first_run: bool
) -> None:
    """Render a compact heading for login or first-administrator setup."""

    title = (
        translate("Set up the first account", "Thiết lập tài khoản đầu tiên")
        if first_run
        else translate("Welcome back", "Chào mừng trở lại")
    )
    detail = (
        translate(
            "Create an administrator account to start using the workspace.",
            "Tạo tài khoản quản trị để bắt đầu sử dụng hệ thống.",
        )
        if first_run
        else translate(
            "Sign in to continue to your research workspace.",
            "Đăng nhập để tiếp tục vào không gian làm việc của bạn.",
        )
    )
    st.markdown(
        f"""
        <div class="auth-card-heading">
            <div class="auth-card-icon" aria-hidden="true">✦</div>
            <div class="auth-card-eyebrow">NeuroScreen</div>
            <h2>{title}</h2>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(
    version: str,
    feature_count: int,
    threshold: float,
    translate: Callable[[str, str], str],
) -> None:
    """Show the active artifact context without crowding the main page."""

    st.sidebar.markdown(
        f"""
        <div class="sidebar-status">
            <div class="sidebar-status-label">{translate("Active artifact", "Artifact đang dùng")}</div>
            <div class="sidebar-status-value">{version}</div>
            <div class="sidebar-status-meta">{feature_count} {translate("features", "đặc trưng")} · {translate("threshold", "ngưỡng")} {threshold:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
