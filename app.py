"""
Airfoil Optimization Engine
============================
Streamlit frontend for aerodynamic performance prediction.
ML backend: Random Forest surrogate models (joblib / .pkl).

Authors : tanmay
          shyam
          ann
          gargi

Status  : Frontend ready
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from optimizer import run_optimization

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
# Loads the three trained Random Forest models from disk via joblib.
# Wrapped in try/except so a missing or corrupted .pkl file cannot crash the
# app — loading simply falls back to None, which keeps MODELS_READY False.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        cl_model = joblib.load("models/rf_cl.pkl")
        cd_model = joblib.load("models/rf_cd.pkl")
        cm_model = joblib.load("models/rf_cm.pkl")
        scaler = joblib.load("models/scaler.pkl")
    except Exception:
        cl_model = cd_model = cm_model = scaler = None

    return cl_model, cd_model, cm_model, scaler


cl_model, cd_model, cm_model, scaler = load_models()

MODELS_READY = (
    cl_model is not None
    and cd_model is not None
    and cm_model is not None
    and scaler is not None
)

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

if "optimization_results" not in st.session_state:
    st.session_state["optimization_results"] = None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(features: dict) -> dict:
    """
    Returns predicted aerodynamic coefficients for given input features.

    Model integration workflow: when MODELS_READY is True, this function
    builds a (1, 6) pandas DataFrame with named columns — matching the
    DataFrame the models were trained on — validates it, and runs
    inference with the three Random Forest regressors. Using named columns
    instead of a raw NumPy array avoids scikit-learn's "X does not have
    valid feature names" warning, since the underlying estimators were
    fitted on a DataFrame with these exact column names.
    When models aren't loaded (or validation fails), fixed placeholder
    values are returned so the UI remains fully testable in demo mode.

    Parameters
    ----------
    features : dict
        Keys (6 model inputs, exact training order): reynolds, aoa,
        max_thickness, thickness_location, max_camber, camber_location

    Returns
    -------
    dict with keys: cl, cd, cm, efficiency
        cl, cd, cm are model outputs; efficiency is derived as cl / cd.
    """
    # TRAINING FEATURE ORDER:
    # 1. reynolds  2. aoa  3. max_thickness
    # 4. thickness_location  5. max_camber  6. camber_location
    # WARNING: changing this order will break model predictions —
    # it must match the order used during model training.
    FEATURE_ORDER = [
        "reynolds", "aoa", "max_thickness",
        "thickness_location", "max_camber", "camber_location",
    ]

    if MODELS_READY:
        # Build a named-column DataFrame (matches the training data format)
        feature_df = pd.DataFrame([features], columns=FEATURE_ORDER)

        # Prediction validation — exact columns, correct order, no missing
        # values, and a (1, 6) shape before inference.
        valid = (
            list(feature_df.columns) == FEATURE_ORDER
            and not feature_df.isnull().values.any()
            and feature_df.shape == (1, 6)
        )
        if not valid:
            
            st.error("Invalid input data. Please check the input parameters.")
        else:
            feature_scaled = scaler.transform(feature_df.to_numpy())

            cl = float(cl_model.predict(feature_scaled)[0])
            cd = float(cd_model.predict(feature_scaled)[0])
            cm = float(cm_model.predict(feature_scaled)[0])

            efficiency = cl / cd if cd != 0 else 0.0

            return {
                "cl": cl,
                "cd": cd,
                "cm": cm,
                "efficiency": efficiency,
            }

    # Placeholder values — used in demo mode or if validation fails
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

    # ── Synchronized slider + number input helper ──────────────────────────
    # Keeps a slider and a numeric input mirroring the same value. Streamlit
    # ignores the `value=` argument for a widget once its `key` already has
    # an entry in session_state, so synchronization must be done explicitly:
    # each on_change callback writes the new value into both the shared key
    # AND the other widget's own key before the next rerun renders it.
    def _sync_slider_and_number(label, min_value, max_value, default, step, fmt, key):
        slider_key = f"{key}_slider"
        number_key = f"{key}_number"

        if key not in st.session_state:
            st.session_state[key] = default
            st.session_state[slider_key] = default
            st.session_state[number_key] = default

        def _on_slider_change():
            st.session_state[key] = st.session_state[slider_key]
            st.session_state[number_key] = st.session_state[slider_key]

        def _on_number_change():
            st.session_state[key] = st.session_state[number_key]
            st.session_state[slider_key] = st.session_state[number_key]

        col_slider, col_number = st.columns([2, 1])
        with col_slider:
            st.slider(
                label, min_value=min_value, max_value=max_value,
                step=step, key=slider_key, on_change=_on_slider_change,
            )
        with col_number:
            st.number_input(
                label, min_value=min_value, max_value=max_value,
                step=step, format=fmt, key=number_key, on_change=_on_number_change,
                label_visibility="collapsed",
            )
        return st.session_state[key]

    aoa = _sync_slider_and_number(
        "Angle of Attack (°)", -10.0, 20.0, 4.0, 0.5, "%.1f", "aoa"
    )

    st.subheader("Geometry")
    max_thickness = _sync_slider_and_number(
        "Max Thickness (% chord)", 4.0, 24.0, 12.0, 0.5, "%.1f", "max_thickness"
    )
    thickness_location = _sync_slider_and_number(
        "Thickness Location (% chord)", 20.0, 60.0, 30.0, 1.0, "%.0f", "thickness_location"
    )
    max_camber = _sync_slider_and_number(
        "Max Camber (% chord)", 0.0, 10.0, 2.0, 0.1, "%.1f", "max_camber"
    )
    camber_location = _sync_slider_and_number(
        "Camber Location (% chord)", 10.0, 70.0, 40.0, 1.0, "%.0f", "camber_location"
    )

    st.divider()
    predict_btn = st.button("Run Prediction", width="stretch", type="primary")

    # Clean, user-facing status — never exposes load exceptions.
    # Diagnostic check: all three models (rf_cl, rf_cd, rf_cm) must be
    # loaded for MODELS_READY to be True (see MODEL LOADER section above).
    if MODELS_READY:
        st.success("Random Forest models loaded successfully")
        st.caption("**Model:** Random Forest Regressor  \n**Outputs:** Cl, Cd, Cm")
    else:
        st.error("Model loading error")


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
    c1.metric(
        "Lift Coefficient (Cl)", f"{cl:.4f}" if _has_results else "—",
        help="Lift Coefficient (Cl) indicates how effectively the airfoil generates lift. "
             "Higher values generally correspond to greater lifting capability."
    )
    c2.metric(
        "Drag Coefficient (Cd)", f"{cd:.4f}" if _has_results else "—",
        help="Drag Coefficient (Cd) indicates the aerodynamic resistance experienced by the "
             "airfoil. Lower values generally correspond to better aerodynamic performance."
    )
    c3.metric(
        "Moment Coefficient (Cm)", f"{cm_out:.4f}" if _has_results else "—",
        help="Moment Coefficient (Cm) measures the pitching moment generated by the airfoil. "
             "A positive Cm indicates a tendency for the nose to pitch upward. A negative Cm "
             "indicates a tendency for the nose to pitch downward. Values closer to zero "
             "generally indicate more balanced aerodynamic behavior."
    )
    c4.metric(
        "Aerodynamic Efficiency (Cl/Cd)", f"{efficiency:.2f}" if _has_results else "—",
        help="Aerodynamic Efficiency is the ratio of Lift Coefficient (Cl) to Drag Coefficient "
             "(Cd). Higher values indicate the airfoil generates more lift for a given amount "
             "of drag. A larger Cl/Cd ratio generally represents a more aerodynamically "
             "efficient design."
    )

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
    st.plotly_chart(airfoil_fig, width="stretch")

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

    # Coefficient bar chart — Cl and Cd only (Cm has its own card; different scale)
    with col_bar:
        bar_fig = go.Figure(go.Bar(
            x=["Cl (Lift)", "Cd (Drag)"],
            y=[cl, cd],
            marker_color=["#4C9BE8", "#E8874C"],
            text=[f"{v:.4f}" for v in [cl, cd]],
            textposition="outside",
            textfont=dict(color="#6B7280"),
            width=0.4,
        ))
        bar_fig.update_layout(
            title=dict(text="Aerodynamic Coefficients", font=dict(color="#6B7280")),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickfont=dict(color="#6B7280")),
            yaxis=dict(zeroline=True, gridcolor="#e0e0e0", tickfont=dict(color="#6B7280")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(bar_fig, width="stretch")

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

    # ── Moment Coefficient card ──────────────────────────────────────────────
    if abs(cm_out) < 0.01:
        cm_interpretation = "Near-zero pitching moment"
    elif cm_out > 0:
        cm_interpretation = "Positive pitching moment"
    else:
        cm_interpretation = "Negative pitching moment"

    st.markdown(f"""
    <div style="background:white; border:1px solid #e0e0e0; border-radius:8px;
                padding:24px; display:flex; flex-direction:column;
                justify-content:center; align-items:center; text-align:center;">
        <div style="font-size:14px; color:#6B7280; font-weight:600; margin-bottom:12px;">
            Moment Coefficient (Cm)
        </div>
        <div style="font-size:48px; font-weight:800; color:#111827; line-height:1.1;">
            {cm_out:.4f}
        </div>
        <div style="font-size:16px; font-weight:600; color:#374151; margin-top:14px;">
            {cm_interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("ℹ️ Positive Cm: nose pitches upward. Negative Cm: nose pitches downward.")

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

        with st.spinner("Running genetic algorithm optimization..."):

            st.session_state["optimization_results"] = run_optimization(
                cl_model=cl_model,
                cd_model=cd_model,
                cm_model=cm_model,
                scaler=scaler,
                reynolds=reynolds,
                aoa=aoa,
                objective=objective,
                iterations=n_iterations,
                constraint_cl=constraint_cl,
                constraint_cd=constraint_cd,
            )

    if st.session_state["optimization_results"] is not None:

        result = st.session_state["optimization_results"]

        st.success("Optimization completed successfully!")

        st.subheader("Optimized Geometry")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Max Thickness",
            f"{result['max_thickness']:.2f}%"
        )

        c2.metric(
            "Thickness Location",
            f"{result['thickness_location']:.2f}%"
        )

        c3.metric(
            "Max Camber",
            f"{result['max_camber']:.2f}%"
        )

        c4.metric(
            "Camber Location",
            f"{result['camber_location']:.2f}%"
        )

        st.divider()

        st.subheader("Predicted Performance")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("Cl", f"{result['cl']:.4f}")
        p2.metric("Cd", f"{result['cd']:.4f}")
        p3.metric("Cm", f"{result['cm']:.4f}")
        p4.metric("Cl/Cd", f"{result['efficiency']:.2f}")

        st.divider()

        st.subheader("Convergence")

        convergence = pd.DataFrame(
            {
                "Generation": range(
                    1,
                    len(result["fitness_history"]) + 1
                ),
                "Fitness": result["fitness_history"],
            }
        )

        st.line_chart(
            convergence.set_index("Generation")
        )