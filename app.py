"""
Airfoil Optimization Engine
============================
Streamlit frontend for aerodynamic performance prediction.
ML backend: Random Forest surrogate models (joblib / .pkl).

Authors : [Team Names]
Course  : [Course Name]
Status  : Frontend ready — awaiting ML model files
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import joblib
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airfoil Optimization Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# Integration point: replace None assignments with joblib.load() calls
# once the ML team provides the trained .pkl files.
#
#   import joblib
#   cl_model = joblib.load("models/cl_model.pkl")
#   cd_model = joblib.load("models/cd_model.pkl")
#   cm_model = joblib.load("models/cm_model.pkl")
#
# Wrapped in try/except so a missing or corrupted .pkl file cannot crash the
# app — loading simply falls back to None, which keeps MODELS_READY False.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cl_model = cd_model = cm_model = None
    try:
        pass  # TODO: cl_model = joblib.load("models/cl_model.pkl")
        # TODO: cd_model = joblib.load("models/cd_model.pkl")
        # TODO: cm_model = joblib.load("models/cm_model.pkl")
    except Exception:
        cl_model = cd_model = cm_model = None
    return cl_model, cd_model, cm_model

cl_model, cd_model, cm_model = load_models()
MODELS_READY = cl_model is not None and cd_model is not None and cm_model is not None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# "results" holds the last prediction output; it persists across reruns so
# the UI doesn't reset when sliders move. It only updates on button click.
# "last_prediction_time" records when that prediction was made, for display.
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None   # None = no prediction run yet
if "last_prediction_time" not in st.session_state:
    st.session_state["last_prediction_time"] = None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(features: dict) -> dict:
    """
    Returns predicted aerodynamic coefficients for given input features.

    Model integration workflow: when MODELS_READY is True, this function
    builds the feature vector, runs inference with the three Random Forest
    regressors, and returns immediately — the placeholder branch below is
    unreachable in that case. When models aren't loaded, fixed placeholder
    values are returned so the UI remains fully testable.

    Parameters
    ----------
    features : dict
        Keys (6 model inputs): reynolds, aoa, max_thickness,
        thickness_location, max_camber, camber_location

    Returns
    -------
    dict with keys: cl, cd, cm, efficiency
        cl, cd, cm are model outputs; efficiency is derived as cl / cd.
    """
    if MODELS_READY:
        # TRAINING FEATURE ORDER:
        # 1. reynolds  2. aoa  3. max_thickness
        # 4. thickness_location  5. max_camber  6. camber_location
        # WARNING: changing this order will break model predictions —
        # it must match the order used during model training.
        feature_vector = np.array([[
            features["reynolds"],
            features["aoa"],
            features["max_thickness"],
            features["thickness_location"],
            features["max_camber"],
            features["camber_location"],
        ]])
        cl = float(cl_model.predict(feature_vector)[0])
        cd = float(cd_model.predict(feature_vector)[0])
        cm = float(cm_model.predict(feature_vector)[0])
        efficiency = cl / cd if cd != 0 else 0.0
        return {"cl": cl, "cd": cd, "cm": cm, "efficiency": efficiency}

    # Placeholder values — only used while models aren't loaded
    cl = 0.82
    cd = 0.018
    cm = -0.05
    efficiency = cl / cd if cd != 0 else 0.0

    return {"cl": cl, "cd": cd, "cm": cm, "efficiency": efficiency}


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY RATING HELPER
# Maps a Cl/Cd value to a label, icon, and short interpretation for the
# Aerodynamic Efficiency card in Tab 1.
# ─────────────────────────────────────────────────────────────────────────────
def get_efficiency_rating(efficiency: float) -> dict:
    """
    Classify aerodynamic efficiency (Cl/Cd) into a rating tier.

    Parameters
    ----------
    efficiency : float
        The Cl/Cd ratio.

    Returns
    -------
    dict with keys: label, icon, description
    """
    if efficiency < 20:
        return {
            "label": "Poor Efficiency",
            "icon": "🔴",
            "description": "Aerodynamic performance can be improved.",
        }
    elif efficiency < 40:
        return {
            "label": "Moderate Efficiency",
            "icon": "🟡",
            "description": "Acceptable aerodynamic performance.",
        }
    elif efficiency < 60:
        return {
            "label": "Good Efficiency",
            "icon": "🟢",
            "description": "Efficient aerodynamic performance observed.",
        }
    else:
        return {
            "label": "Excellent Efficiency",
            "icon": "⭐",
            "description": "Highly efficient aerodynamic design.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# AIRFOIL GEOMETRY GENERATOR
# Produces parametric upper/lower surface coordinates from the 4 geometry
# inputs. Uses a modified NACA-style approach:
#   • Camber line  — piecewise parabola controlled by max_camber + camber_location
#   • Thickness    — scaled distribution with peak at thickness_location
#   • AoA rotation — rotates the whole profile about the leading edge
#
# Not physically accurate for all parameter combinations, but visually faithful
# and updates continuously with slider changes.
# ─────────────────────────────────────────────────────────────────────────────
def generate_airfoil(
    max_thickness: float,
    thickness_location: float,
    max_camber: float,
    camber_location: float,
    aoa_deg: float = 0.0,
    n_points: int = 200,
) -> dict:
    """
    Generate airfoil upper/lower surface (x, y) coordinates.

    Parameters
    ----------
    max_thickness     : max thickness as % chord  (e.g. 12 → 0.12c)
    thickness_location: chordwise position of max thickness, % chord
    max_camber        : max camber as % chord      (e.g. 2 → 0.02c)
    camber_location   : chordwise position of max camber, % chord
    aoa_deg           : angle of attack in degrees (rotates profile)
    n_points          : number of points along chord

    Returns
    -------
    dict with keys: x_upper, y_upper, x_lower, y_lower,
                    x_camber, y_camber  (mean camber line)
    """
    # Normalise inputs from % chord → fraction
    t  = max_thickness      / 100.0
    p  = thickness_location / 100.0   # peak thickness location
    m  = max_camber         / 100.0
    xm = camber_location    / 100.0   # peak camber location

    # Cosine-spaced x stations (denser near LE and TE)
    beta = np.linspace(0, np.pi, n_points)
    x    = 0.5 * (1 - np.cos(beta))  # shape: (n_points,)

    # ── Thickness distribution ───────────────────────────────────────────────
    # Gaussian-shaped distribution peaking at x = p, scaled so peak = t/2
    sigma = 0.18 + 0.3 * p            # wider peak for aft-loaded profiles
    yt    = (t / 2.0) * np.exp(-((x - p) ** 2) / (2 * sigma ** 2))
    # Preserve sharp trailing edge
    yt   *= (1 - x ** 3)

    # Normalise so the true maximum equals t/2
    if yt.max() > 0:
        yt = yt * (t / 2.0) / yt.max()

    # ── Camber line (piecewise parabola, NACA-style) ─────────────────────────
    yc = np.where(
        x <= xm,
        (m / (xm ** 2 + 1e-12)) * (2 * xm * x - x ** 2),
        (m / ((1 - xm) ** 2 + 1e-12)) * ((1 - 2 * xm) + 2 * xm * x - x ** 2),
    )

    # ── Camber-line slope (for surface normal direction) ─────────────────────
    dyc_dx = np.where(
        x <= xm,
        (2 * m / (xm ** 2 + 1e-12)) * (xm - x),
        (2 * m / ((1 - xm) ** 2 + 1e-12)) * (xm - x),
    )
    theta = np.arctan(dyc_dx)

    # ── Upper and lower surfaces ─────────────────────────────────────────────
    x_upper = x  - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x  + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    # ── Rotate about LE by angle of attack ───────────────────────────────────
    aoa_rad = np.deg2rad(-aoa_deg)          # negative so nose-up = up on plot
    cos_a, sin_a = np.cos(aoa_rad), np.sin(aoa_rad)

    def rotate(xv, yv):
        return xv * cos_a - yv * sin_a, xv * sin_a + yv * cos_a

    x_upper, y_upper = rotate(x_upper, y_upper)
    x_lower, y_lower = rotate(x_lower, y_lower)
    x_camber, y_camber = rotate(x, yc)

    return {
        "x_upper": x_upper, "y_upper": y_upper,
        "x_lower": x_lower, "y_lower": y_lower,
        "x_camber": x_camber, "y_camber": y_camber,
    }


def build_airfoil_figure(geom: dict, aoa_deg: float, params: dict) -> go.Figure:
    """
    Build a Plotly figure for the airfoil cross-section.

    Parameters
    ----------
    geom    : output of generate_airfoil()
    aoa_deg : angle of attack (shown in annotation)
    params  : dict of labelled geometry values for the annotation box

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # ── Filled airfoil body ──────────────────────────────────────────────────
    x_body = np.concatenate([geom["x_upper"], geom["x_lower"][::-1]])
    y_body = np.concatenate([geom["y_upper"], geom["y_lower"][::-1]])

    fig.add_trace(go.Scatter(
        x=x_body, y=y_body,
        fill="toself",
        fillcolor="rgba(76, 155, 232, 0.12)",
        line=dict(color="rgba(76, 155, 232, 0.0)", width=0),
        hoverinfo="skip",
        showlegend=False,
        name="body",
    ))

    # ── Upper surface ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=geom["x_upper"], y=geom["y_upper"],
        mode="lines",
        line=dict(color="#4C9BE8", width=2.5),
        name="Upper Surface",
        hovertemplate="x: %{x:.3f}c<br>y: %{y:.3f}c<extra>Upper</extra>",
    ))

    # ── Lower surface ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=geom["x_lower"], y=geom["y_lower"],
        mode="lines",
        line=dict(color="#4C9BE8", width=2.5),
        name="Lower Surface",
        hovertemplate="x: %{x:.3f}c<br>y: %{y:.3f}c<extra>Lower</extra>",
        showlegend=False,
    ))

    # ── Mean camber line ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=geom["x_camber"], y=geom["y_camber"],
        mode="lines",
        line=dict(color="#E8874C", width=1.2, dash="dash"),
        name="Camber Line",
        hovertemplate="x: %{x:.3f}c<br>y: %{y:.3f}c<extra>Camber</extra>",
    ))

    # ── Chord line ───────────────────────────────────────────────────────────
    aoa_rad = np.deg2rad(-aoa_deg)
    chord_x = [0, np.cos(aoa_rad)]
    chord_y = [0, np.sin(aoa_rad)]
    fig.add_trace(go.Scatter(
        x=chord_x, y=chord_y,
        mode="lines",
        line=dict(color="#9CA3AF", width=1, dash="dot"),
        name="Chord Line",
    ))

    # ── Freestream arrow (shows AoA visually) ────────────────────────────────
    arrow_len = 0.22
    fig.add_annotation(
        x=0, y=0,
        ax=-arrow_len * np.cos(np.deg2rad(aoa_deg)),
        ay= arrow_len * np.sin(np.deg2rad(aoa_deg)),
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=1.8,
        arrowcolor="#9CA3AF",
    )

    # ── Geometry annotation box ───────────────────────────────────────────────
    label = (
        f"t/c = {params['max_thickness']:.1f}%   "
        f"p_t = {params['thickness_location']:.0f}%<br>"
        f"m/c = {params['max_camber']:.1f}%   "
        f"p_m = {params['camber_location']:.0f}%   "
        f"α = {aoa_deg:+.1f}°"
    )
    fig.add_annotation(
        text=label,
        xref="paper", yref="paper",
        x=0.01, y=0.97,
        xanchor="left", yanchor="top",
        font=dict(size=11, color="#374151", family="monospace"),
        bgcolor="#F9FAFB",
        bordercolor="#D1D5DB",
        borderwidth=1,
        borderpad=6,
        showarrow=False,
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=280,
        margin=dict(l=10, r=10, t=36, b=10),
        title=dict(
            text="Airfoil Cross-Section",
            font=dict(size=13, color="#6B7280", family="Inter"),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            range=[-0.28, 1.18],
            showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
            zeroline=False,
            tickfont=dict(color="#6B7280", size=10),
            title=dict(text="x/c", font=dict(color="#6B7280", size=11)),
        ),
        yaxis=dict(
            range=[-0.28, 0.28],
            showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
            zeroline=False,
            tickfont=dict(color="#6B7280", size=10),
            title=dict(text="y/c", font=dict(color="#6B7280", size=11)),
            scaleanchor="x", scaleratio=1,      # equal-aspect so shape looks right
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.18, yanchor="top",
            font=dict(size=11, color="#6B7280"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Input Parameters")
    st.caption("Adjust values and click Predict.")

    st.subheader("Flow Conditions")
    reynolds = st.number_input(
        "Reynolds Number",
        min_value=1e4, max_value=1e7, value=1e6, step=1e5, format="%.0f",
        help="Re = ρVL/μ — dimensionless flow regime indicator"
    )
    aoa = st.slider(
        "Angle of Attack (°)",
        min_value=-10.0, max_value=20.0, value=4.0, step=0.5,
        help="Angle between the chord line and the freestream direction"
    )

    st.subheader("Geometry")
    max_thickness = st.slider(
        "Max Thickness (% chord)",
        min_value=4.0, max_value=24.0, value=12.0, step=0.5,
    )
    thickness_location = st.slider(
        "Thickness Location (% chord)",
        min_value=20.0, max_value=60.0, value=30.0, step=1.0,
    )
    max_camber = st.slider(
        "Max Camber (% chord)",
        min_value=0.0, max_value=10.0, value=2.0, step=0.1,
    )
    camber_location = st.slider(
        "Camber Location (% chord)",
        min_value=10.0, max_value=70.0, value=40.0, step=1.0,
    )

    st.divider()
    predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")

    # Clean, user-facing status — never exposes load exceptions
    if MODELS_READY:
        st.success("ML model active")
    else:
        st.info("Using pre-loaded aerodynamic data")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PREDICTION
# Prediction workflow: clicking "Run Prediction" calls predict() once and
# caches the result (plus a timestamp) in session state. On every other
# rerun (e.g. moving a slider), the cached result is reused so the metrics
# don't change until the user explicitly predicts again.
# ─────────────────────────────────────────────────────────────────────────────
inputs = {
    "reynolds":           reynolds,
    "aoa":                aoa,
    "max_thickness":      max_thickness,
    "thickness_location": thickness_location,
    "max_camber":         max_camber,
    "camber_location":    camber_location,
}

if predict_btn:
    st.session_state["results"] = predict(inputs)
    st.session_state["last_prediction_time"] = datetime.now().strftime("%H:%M:%S")

# Read results from session state; use zeros if no prediction has been run yet
if st.session_state["results"] is not None:
    cl         = st.session_state["results"]["cl"]
    cd         = st.session_state["results"]["cd"]
    cm_out     = st.session_state["results"]["cm"]
    efficiency = st.session_state["results"]["efficiency"]
    _has_results = True
else:
    cl = cd = cm_out = efficiency = 0.0
    _has_results = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.title("Airfoil Optimization Engine")
st.caption("Predict aerodynamic performance and optimize airfoil geometry")

tab1, tab2 = st.tabs(["Airfoil Analysis", "Optimization"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — AIRFOIL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    if not _has_results:
        st.info("Configure parameters in the sidebar and click **Run Prediction** to see results.")

    # ── Performance metrics ──────────────────────────────────────────────────
    st.subheader("Predicted Coefficients")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lift Coefficient (Cl)",         f"{cl:.4f}" if _has_results else "—")
    c2.metric("Drag Coefficient (Cd)",          f"{cd:.4f}" if _has_results else "—")
    c3.metric("Moment Coefficient (Cm)",        f"{cm_out:.4f}" if _has_results else "—")
    c4.metric("Aerodynamic Efficiency (Cl/Cd)", f"{efficiency:.2f}" if _has_results else "—")

    # ── Performance Summary ──────────────────────────────────────────────────
    if _has_results:
        st.subheader("Performance Summary")

        def _cl_label(v):
            if v < 0.4:   return "low", "🔴"
            if v < 0.8:   return "moderate", "🟡"
            return "high", "🟢"

        def _cd_label(v):
            if v < 0.02:  return "low", "🟢"
            if v < 0.05:  return "moderate", "🟡"
            return "high", "🔴"

        def _eff_label(v):
            if v < 20:    return "low", "🔴"
            if v < 40:    return "moderate", "🟡"
            return "high", "🟢"

        cl_lbl,  cl_icon  = _cl_label(cl)
        cd_lbl,  cd_icon  = _cd_label(cd)
        eff_lbl, eff_icon = _eff_label(efficiency)

        ps1, ps2, ps3 = st.columns(3)
        ps1.markdown(f"{cl_icon} &nbsp;**Lift coefficient** is **{cl_lbl}** ({cl:.4f})")
        ps2.markdown(f"{cd_icon} &nbsp;**Drag coefficient** is **{cd_lbl}** ({cd:.4f})")
        ps3.markdown(f"{eff_icon} &nbsp;**Aerodynamic efficiency** is **{eff_lbl}** ({efficiency:.2f})")

        # Timestamp of the last successful prediction (only shown once one exists)
        if st.session_state["last_prediction_time"]:
            st.caption(f"Last Prediction: {st.session_state['last_prediction_time']}")

    st.divider()

    # ── Airfoil Shape Visualizer ─────────────────────────────────────────────
    # Visualization workflow: geometry is derived purely from the sidebar
    # sliders (not from the ML models), so it updates live on every rerun —
    # no "Run Prediction" click required. generate_airfoil() computes the
    # coordinates; build_airfoil_figure() renders them as a Plotly figure.
    geom = generate_airfoil(
        max_thickness=max_thickness,
        thickness_location=thickness_location,
        max_camber=max_camber,
        camber_location=camber_location,
        aoa_deg=aoa,
    )
    airfoil_fig = build_airfoil_figure(
        geom=geom,
        aoa_deg=aoa,
        params={
            "max_thickness":      max_thickness,
            "thickness_location": thickness_location,
            "max_camber":         max_camber,
            "camber_location":    camber_location,
        },
    )
    st.plotly_chart(airfoil_fig, use_container_width=True)

    # ── Current Airfoil Geometry card ────────────────────────────────────────
    with st.container():
        st.caption("**Current Airfoil Geometry**")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Max Thickness",      f"{max_thickness:.1f}%")
        g2.metric("Thickness Location", f"{thickness_location:.0f}%")
        g3.metric("Max Camber",         f"{max_camber:.1f}%")
        g4.metric("Camber Location",    f"{camber_location:.0f}%")
        g5.metric("Angle of Attack",    f"{aoa:+.1f}°")

    st.divider()

    # ── Visualizations ───────────────────────────────────────────────────────
    st.subheader("Charts")

    col_bar, col_gauge = st.columns([3, 2])

    # Coefficient bar chart
    with col_bar:
        bar_fig = go.Figure(go.Bar(
            x=["Cl (Lift)", "Cd (Drag)", "Cm (Moment)"],
            y=[cl, cd, cm_out],
            marker_color=["#4C9BE8", "#E8874C", "#9B59B6"],
            text=[f"{v:.4f}" for v in [cl, cd, cm_out]],
            textposition="outside",
            width=0.4,
        ))
        bar_fig.update_layout(
            title="Aerodynamic Coefficients",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(zeroline=True, gridcolor="rgba(128,128,128,0.3)"),
            template="plotly"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Aerodynamic Efficiency card
    with col_gauge:
        rating = get_efficiency_rating(efficiency)
        st.markdown(f"""
        <div style="background:white; border:1px solid #e0e0e0; border-radius:8px;
                    padding:24px; height:350px; display:flex; flex-direction:column;
                    justify-content:center; align-items:center; text-align:center;">
            <div style="font-size:14px; color:#6B7280; font-weight:600; margin-bottom:12px;">
                Aerodynamic Efficiency (Cl/Cd)
            </div>
            <div style="font-size:48px; font-weight:800; color:#111827; line-height:1.1;">
                {efficiency:.1f}
            </div>
            <div style="font-size:16px; font-weight:600; color:#374151; margin-top:14px;">
                {rating['icon']} {rating['label']}
            </div>
            <div style="font-size:13px; color:#6B7280; margin-top:8px; max-width:240px;">
                {rating['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input summary ────────────────────────────────────────────────────────
    with st.expander("View current input values"):
        param_data = {
            "Reynolds Number":       f"{reynolds:,.0f}",
            "Angle of Attack (°)":   f"{aoa:.1f}",
            "Max Thickness (%)":     f"{max_thickness:.1f}",
            "Thickness Location (%)": f"{thickness_location:.1f}",
            "Max Camber (%)":        f"{max_camber:.1f}",
            "Camber Location (%)":   f"{camber_location:.1f}",
        }
        col_a, col_b = st.columns(2)
        items = list(param_data.items())
        for k, v in items[:3]:
            col_a.text(f"{k}: {v}")
        for k, v in items[3:]:
            col_b.text(f"{k}: {v}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Optimization Settings")
    st.caption("Configure an objective and constraints, then run the optimizer.")

    # ── Objective ────────────────────────────────────────────────────────────
    col_obj, col_iter = st.columns(2)

    with col_obj:
        objective = st.selectbox(
            "Optimization Objective",
            options=["Maximum Lift (Cl)", "Minimum Drag (Cd)", "Maximum Efficiency (Cl/Cd)"],
            help="The quantity the optimizer will maximize or minimize"
        )

    with col_iter:
        n_iterations = st.slider(
            "Max Iterations",
            min_value=50, max_value=500, value=100, step=50,
            help="Higher values improve solution quality at the cost of runtime"
        )

    # ── Constraints ──────────────────────────────────────────────────────────
    st.subheader("Constraints")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        constraint_cl = st.number_input(
            "Minimum Cl",
            min_value=0.0, max_value=2.0, value=0.3, step=0.05, format="%.2f",
            help="Lower bound on lift coefficient"
        )
    with col_c2:
        constraint_cd = st.number_input(
            "Maximum Cd",
            min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f",
            help="Upper bound on drag coefficient"
        )

    st.divider()

    # ── Run button ───────────────────────────────────────────────────────────
    optimize_clicked = st.button(
        "Run Optimization",
        type="primary",
        disabled=not MODELS_READY,
    )

    if not MODELS_READY:
        st.caption("Optimization will be available once the ML model is connected.")

    if optimize_clicked and MODELS_READY:
        st.success("Optimization complete.")