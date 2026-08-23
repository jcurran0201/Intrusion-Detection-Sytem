"""
Phase 11 — Streamlit dashboard

Panels:
  - PCAP Upload      — upload file → POST /predict → show results
  - Live Capture     — start / stop live monitoring → poll /live/status
  - Flow Table       — per-flow predictions with color-coded alert tier badges
  - Distribution     — attack class bar chart
  - Alert Tier Pie   — HIGH / MEDIUM / LOW breakdown

Run:
  streamlit run dashboard/app.py
  (ensure the FastAPI server is running on localhost:8000 first)
"""

import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"

TIER_COLORS = {
    "HIGH":   "#ff6b6b",
    "MEDIUM": "#ffd43b",
    "LOW":    "#69db7c",
}

ATTACK_PALETTE = [
    "#4fc3f7", "#ff6b6b", "#69db7c", "#ffd43b",
    "#da77f2", "#ff922b", "#74c0fc", "#f06595",
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS — Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .tier-HIGH   { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b;
                 border-radius:4px; padding:2px 8px; font-weight:700; }
  .tier-MEDIUM { background:#ffd43b22; color:#ffd43b; border:1px solid #ffd43b;
                 border-radius:4px; padding:2px 8px; font-weight:700; }
  .tier-LOW    { background:#69db7c22; color:#69db7c; border:1px solid #69db7c;
                 border-radius:4px; padding:2px 8px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Intrusion Detection System")
st.caption("Behavioral flow-level classification — upload a PCAP or monitor live traffic.")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_base = st.text_input("API URL", value=API_BASE)
    st.divider()

    st.subheader("🎯 Alert Thresholds")
    high_thresh   = st.slider("HIGH threshold",   0.50, 1.0, 0.85, 0.01)
    medium_thresh = st.slider("MEDIUM threshold", 0.10, 0.85, 0.50, 0.01)
    st.caption("Thresholds shown here are for display only — the model uses fixed tiers.")
    st.divider()

    st.subheader("📡 Live Capture")
    interface      = st.text_input("Network interface", value="en0")
    window_seconds = st.number_input("Rolling window (s)", 10, 300, 30)

# ── helper ────────────────────────────────────────────────────────────────────

def post(endpoint: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{api_base}{endpoint}", timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API. Is `uvicorn api.main:app` running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    return None


def get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{api_base}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    return None


def render_results(data: dict):
    """Render summary metrics, flow table, and charts from an API response."""
    summary = data.get("summary", {})
    flows   = data.get("flows", [])

    if not flows:
        st.warning("No flows returned.")
        return

    # ── metric row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total flows",  summary.get("total_flows", len(flows)))
    col2.metric("🔴 HIGH",      summary.get("HIGH",   0))
    col3.metric("🟡 MEDIUM",    summary.get("MEDIUM", 0))
    col4.metric("🟢 LOW",       summary.get("LOW",    0))

    df = pd.DataFrame(flows)

    # ── charts ────────────────────────────────────────────────────────────────
    chart_col, pie_col = st.columns(2)

    with chart_col:
        st.subheader("Attack Class Distribution")
        class_counts = df["pred_label"].value_counts().reset_index()
        class_counts.columns = ["Attack Class", "Count"]
        fig = px.bar(
            class_counts, x="Attack Class", y="Count",
            color="Attack Class", color_discrete_sequence=ATTACK_PALETTE,
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with pie_col:
        st.subheader("Alert Tier Breakdown")
        tier_counts = df["alert_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_color_map = {t: c for t, c in TIER_COLORS.items()}
        fig2 = px.pie(
            tier_counts, names="Tier", values="Count",
            color="Tier", color_discrete_map=tier_color_map,
            template="plotly_dark", hole=0.45,
        )
        fig2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── attack score timeline ─────────────────────────────────────────────────
    st.subheader("Attack Score per Flow")
    df["flow_id"] = range(len(df))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["flow_id"], y=df["attack_score"],
        mode="markers",
        marker=dict(
            color=df["attack_score"],
            colorscale=[[0,"#69db7c"],[0.5,"#ffd43b"],[1,"#ff6b6b"]],
            cmin=0, cmax=1, size=5, opacity=0.7,
            colorbar=dict(title="Score"),
        ),
        hovertemplate="Flow %{x}<br>Score: %{y:.3f}<br>%{text}",
        text=df["pred_label"],
    ))
    fig3.add_hline(y=0.85, line_dash="dash", line_color="#ff6b6b",
                   annotation_text="HIGH (0.85)")
    fig3.add_hline(y=0.50, line_dash="dash", line_color="#ffd43b",
                   annotation_text="MEDIUM (0.50)")
    fig3.update_layout(
        template="plotly_dark",
        xaxis_title="Flow index",
        yaxis_title="Attack score",
        margin=dict(t=20, b=40),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── flow table ────────────────────────────────────────────────────────────
    st.subheader("Flow-Level Predictions")
    display_cols = ["flow_id", "attack_score", "pred_label", "alert_tier", "action"]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].style.apply(
            lambda row: [
                f"background-color: {TIER_COLORS.get(row['alert_tier'], '')}22"
            ] * len(row),
            axis=1,
        ),
        use_container_width=True,
        height=400,
    )


# ── tab layout ────────────────────────────────────────────────────────────────
tab_upload, tab_live = st.tabs(["📁 PCAP Upload", "📡 Live Capture"])

# ── UPLOAD TAB ────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a PCAP for forensic analysis")
    st.caption(
        "Upload a .pcap or .pcapng file. "
        "CICFlowMeter extracts flow features → model classifies each flow."
    )

    uploaded = st.file_uploader("Choose a PCAP file", type=["pcap", "pcapng"])

    if uploaded:
        if st.button("🔍 Analyze", key="btn_analyze"):
            with st.spinner("Running pipeline (pyshark → CICFlowMeter → model)…"):
                result = post(
                    "/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                )
            if result:
                st.success(f"Analysis complete — {result['summary']['total_flows']} flows classified.")
                render_results(result)

# ── LIVE TAB ──────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("Real-time behavioral monitoring")
    st.caption(
        f"Buffers {window_seconds}s of packets from `{interface}` → "
        "dumps to temp PCAP → runs full pipeline. "
        "Model classifies behavioral anomalies across completed flows every ~30s."
    )

    status_placeholder = st.empty()
    results_placeholder = st.empty()

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button("▶️ Start Live Capture", key="btn_start"):
            res = post(
                "/live/start",
                params={"interface": interface, "window_seconds": window_seconds},
            )
            if res:
                st.success(f"Capture started on `{interface}` — {window_seconds}s window.")

    with col_stop:
        if st.button("⏹ Stop & Classify", key="btn_stop"):
            with st.spinner("Stopping capture and running classification…"):
                res = post("/live/stop")
            if res and res.get("flows"):
                st.success("Live capture results:")
                render_results(res)
            elif res:
                st.info("Capture stopped — no flows to classify yet.")

    # ── auto-refresh status ───────────────────────────────────────────────────
    if st.checkbox("Auto-refresh status every 5s", value=False):
        while True:
            status = get("/live/status")
            if status:
                running = status.get("running", False)
                count   = status.get("result_count", 0)
                msg     = "🟢 Running" if running else "⚪ Idle"
                status_placeholder.info(
                    f"**Live capture:** {msg}  |  Flows ready: {count}"
                )
                if not running and count > 0:
                    res = post("/live/stop")
                    if res:
                        with results_placeholder.container():
                            render_results(res)
                    break
            time.sleep(5)
            st.rerun()
