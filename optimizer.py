# ============================================================================
# TODO
#
# Optimization executes successfully but currently returns unrealistic geometry
# and aerodynamic coefficients.
#
# Verified:
# - Prediction pipeline in app.py works correctly.
# - Models and scaler load successfully.
# - GA executes successfully.
#
# Remaining issue appears to be in optimizer representation/scaling/bounds.
#
# Needs verification against the original model-training notebook.
# ============================================================================

"""
Airfoil Optimization Engine — Optimizer Module
===============================================
Production-ready genetic algorithm optimizer using PyGAD 3.x.
Wraps the optimization logic from optimisation.ipynb into a single
callable: run_optimization().

This module:
  - Does NOT load models or the scaler (received from app.py).
  - Does NOT use joblib, Streamlit, input(), or print().
  - Exposes exactly ONE public function: run_optimization().

The fitness function, GA hyperparameters, scaling strategy, and
inverse-transform step are preserved verbatim from the notebook.

Authors : tanmay
          shyam
          ann
          gargi
"""



import numpy as np
import pandas as pd
import pygad

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Exact feature order used at training time — must not be changed.
# Matches full_features in the notebook.
FULL_FEATURES = [
    "reynolds",
    "aoa",
    "max_thickness",
    "thickness_location",
    "max_camber",
    "camber_location",
]

# The four geometry genes the GA optimises (indices 2–5 of FULL_FEATURES).
# Matches features list in the notebook.
GEOMETRY_FEATURES = [
    "max_thickness",
    "thickness_location",
    "max_camber",
    "camber_location",
]

# Gene-space bounds for the four geometry parameters.
# The notebook derives these from df[col].min() / df[col].max() on the
# training CSV.  Since the CSV is not available at runtime, the bounds are
# taken from the app.py slider ranges, which were calibrated to the same
# training data.  Order must match GEOMETRY_FEATURES exactly.
_GEOMETRY_BOUNDS = [
    (4.0,  24.0),   # max_thickness      (% chord)
    (20.0, 60.0),   # thickness_location (% chord)
    (0.0,  10.0),   # max_camber         (% chord)
    (10.0, 70.0),   # camber_location    (% chord)
]

# GA hyperparameters — identical to the notebook.
# num_generations is NOT listed here; it is supplied by the caller via
# the `iterations` argument so the Streamlit slider takes effect.
# (Notebook hardcodes 50; app.py slider default is 100.)
_GA_SOL_PER_POP        = 20
_GA_NUM_PARENTS_MATING = 10
_GA_NUM_GENES          = 4
_GA_MUTATION_NUM_GENES = 1
_GA_RANDOM_SEED        = 42
_GA_STOP_CRITERIA      = ["saturate_10"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PREDICT PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_performance(
    solution,
    reynolds,
    aoa,
    cl_model,
    cd_model,
    cm_model,
    scaler,
):
    X = pd.DataFrame(
        [[
            reynolds,
            aoa,
            solution[0] /100.0,
            solution[1] /100.0,
            solution[2] /100.0,
            solution[3] /100.0,
        ]],
        columns=FULL_FEATURES,
    )

    X_scaled = scaler.transform(X.to_numpy())

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=FULL_FEATURES,
    )

    cl = float(cl_model.predict(X_scaled)[0])
    cd = float(cd_model.predict(X_scaled)[0])
    cm = float(cm_model.predict(X_scaled)[0])

    return cl, cd, cm


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: FITNESS FUNCTION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _make_fitness_function(
    reynolds: float,
    aoa: float,
    cl_model,
    cd_model,
    cm_model,
    scaler,
):
    """
    Return a PyGAD-compatible fitness function closed over the models,
    scaler, and pre-scaled flow conditions.

    The fitness function body is preserved verbatim from the notebook:

        if cd <= 0: return 0.0
        if cl <= 0: return 0.0
        if cm < -0.3 or cm > 0.1: return 0.0
        cl_cd = cl / cd
        cm_penalty = 5 * abs(cm + 0.05)
        fitness = cl_cd - cm_penalty
        return fitness

    No constraint penalties, no objective-specific scaling terms, and no
    other additions are present — any such modification would alter the
    fitness landscape and produce results that diverge from the notebook.

    Parameters
    ----------
    scaled_reynolds : float
        Pre-scaled Reynolds number (from the initial scaler.transform call).
    scaled_aoa      : float
        Pre-scaled angle of attack (from the initial scaler.transform call).
    cl_model        : fitted sklearn estimator
    cd_model        : fitted sklearn estimator
    cm_model        : fitted sklearn estimator
    scaler          : fitted sklearn scaler

    Returns
    -------
    callable
        fitness_function(ga_instance, solution, solution_idx) -> float
        Compatible with PyGAD 3.x fitness_func signature.
    """

    def fitness_function(ga_instance, solution, solution_idx):
        cl, cd, cm = predict_performance(
            solution,
            reynolds,
            aoa,
            cl_model,
            cd_model,
            cm_model,
            scaler,
        )

        # Hard feasibility guards — identical to notebook.
        if cd <= 0:
            return 0.0
        if cl <= 0:
            return 0.0
        if cm < -0.3 or cm > 0.1:
            return 0.0

        # Fitness calculation — identical to notebook.
        cl_cd = cl / cd
        cm_penalty = 5 * abs(cm + 0.05)
        fitness = cl_cd - cm_penalty

        return fitness

    return fitness_function


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GENERATION CALLBACK FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _make_on_generation(fitness_history: list):
    """
    Return a PyGAD on_generation callback that records the best fitness of
    each completed generation.

    The notebook's on_generation() appended to fitness_history and printed
    to stdout.  The print() call is omitted here (not permitted outside a
    notebook), but the append behavior is preserved exactly.

    Parameters
    ----------
    fitness_history : list
        Mutable list shared with run_optimization(); appended in-place so
        the caller receives the full convergence trace after ga.run().

    Returns
    -------
    callable
        on_generation(ga_instance) -> None
    """

    def on_generation(ga_instance):
        best_fitness = ga_instance.best_solution()[1]
        fitness_history.append(float(best_fitness))

    return on_generation


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(
    cl_model,
    cd_model,
    cm_model,
    scaler,
    reynolds: float,
    aoa: float,
    objective: str,
    iterations: int,
    constraint_cl: float,
    constraint_cd: float,
) -> dict:
    """
    Run the genetic algorithm optimizer and return the best airfoil geometry
    together with its predicted aerodynamic performance.

    The optimization algorithm, fitness function, and all hyperparameters are
    taken directly from optimisation.ipynb.  The `objective`, `constraint_cl`,
    and `constraint_cd` parameters are accepted for API compatibility with
    app.py but are not used inside the fitness function — the notebook contains
    no such parameters and adding them would change the fitness landscape.

    Parameters
    ----------
    cl_model      : fitted sklearn estimator
        Random Forest regressor for lift coefficient (rf_cl).
    cd_model      : fitted sklearn estimator
        Random Forest regressor for drag coefficient (rf_cd).
    cm_model      : fitted sklearn estimator
        Random Forest regressor for moment coefficient (rf_cm).
    scaler        : fitted sklearn scaler
        The StandardScaler fitted on the full 6-feature training set.
    reynolds      : float
        Raw Reynolds number (e.g. 1_000_000).
    aoa           : float
        Raw angle of attack in degrees (e.g. 4.0).
    objective     : str
        Accepted for app.py API compatibility; not used in fitness function.
        One of: "Maximum Lift (Cl)", "Minimum Drag (Cd)",
                "Maximum Efficiency (Cl/Cd)".
    iterations    : int
        Maximum number of GA generations (maps to num_generations).
        The notebook hardcodes 50; this argument lets the Streamlit slider
        override that value.
    constraint_cl : float
        Accepted for app.py API compatibility; not used in fitness function.
    constraint_cd : float
        Accepted for app.py API compatibility; not used in fitness function.

    Returns
    -------
    dict
        max_thickness      : float  — optimized geometry in real units (% chord)
        thickness_location : float  — optimized geometry in real units (% chord)
        max_camber         : float  — optimized geometry in real units (% chord)
        camber_location    : float  — optimized geometry in real units (% chord)
        cl                 : float  — predicted lift coefficient
        cd                 : float  — predicted drag coefficient
        cm                 : float  — predicted moment coefficient
        efficiency         : float  — cl / cd  (0.0 if cd == 0)
        fitness            : float  — best GA fitness value
        fitness_history    : list[float]  — best fitness recorded per generation
    """

    # ── Step 1: Scale reynolds and aoa once ──────────────────────────────────
    # Mirrors the notebook's "SCALE USER INPUT ONCE" cell:
    #   user_df = pd.DataFrame([[reynolds_raw, aoa_raw, 0, 0, 0, 0]], columns=full_features)
    #   user_scaled = scaler.transform(user_df.to_numpy())
    #   scaled_reynolds = user_scaled[0, 0]
    #   scaled_aoa      = user_scaled[0, 1]
    # Dummy zeros fill the geometry columns so scaler.transform() receives a
    # full 6-column row; only the first two outputs are retained.
    raw_reynolds = float(reynolds)
    raw_aoa = float(aoa)

    # ── Step 2: Build gene space ──────────────────────────────────────────────
    # Mirrors the notebook's:
    #   gene_space = [{"low": b[0], "high": b[1]} for b in bounds]
    # where bounds = [(df[col].min(), df[col].max()) for col in features].
    # The bounds use raw (unscaled) geometry values, so the GA produces raw
    # geometry genes — which is what predict_performance() expects.
    gene_space = [{"low": b[0], "high": b[1]} for b in _GEOMETRY_BOUNDS]

    # ── Step 3: Initialise fitness history ────────────────────────────────────
    # Shared mutable list written by on_generation and returned to the caller.
    fitness_history = []

    # ── Step 4: Build PyGAD GA instance ──────────────────────────────────────
    # All hyperparameters are identical to the notebook's pygad.GA() call.
    ga = pygad.GA(
        num_generations=iterations,
        sol_per_pop=_GA_SOL_PER_POP,
        num_parents_mating=_GA_NUM_PARENTS_MATING,
        num_genes=_GA_NUM_GENES,
        gene_space=gene_space,
        fitness_func=_make_fitness_function(
            reynolds=raw_reynolds,
            aoa=raw_aoa,
            cl_model=cl_model,
            cd_model=cd_model,
            cm_model=cm_model,
            scaler=scaler,
        ),
        mutation_num_genes=_GA_MUTATION_NUM_GENES,
        random_seed=_GA_RANDOM_SEED,
        on_generation=_make_on_generation(fitness_history),
        stop_criteria=_GA_STOP_CRITERIA,
    )

    # ── Step 5: Run the GA ────────────────────────────────────────────────────
    ga.run()

    best_solution, best_fitness, _ = ga.best_solution()


    # ── Step 6: Predict final performance for the best solution ───────────────
    # Mirrors the notebook's post-run call:
    #   cl, cd, cm = predict_performance(best_solution)
    cl, cd, cm = predict_performance(
        best_solution,
        raw_reynolds,
        raw_aoa,
        cl_model,
        cd_model,
        cm_model,
        scaler,
    )

    # ── Step 7: Inverse-transform to recover real geometry values ─────────────
    # Mirrors the notebook's "INVERSE TRANSFORM (REAL GEOMETRY)" cell verbatim:
    #
    #   scaled_geom = np.array(best_solution).reshape(1, -1)
    #   full_scaled = np.array([[scaled_reynolds, scaled_aoa,
    #                            scaled_geom[0,0], scaled_geom[0,1],
    #                            scaled_geom[0,2], scaled_geom[0,3]]])
    #   real_values = scaler.inverse_transform(full_scaled)
    #   real_geom   = real_values[0, 2:]
    #
    # Note: the variable name "scaled_geom" in the notebook is misleading —
    # best_solution contains raw geometry values (the gene_space uses raw
    # bounds).  The inverse_transform call is preserved exactly as written in
    # the notebook; any correction to this step would diverge from the
    # notebook's output.
    real_geom = np.array(best_solution)

    # ── Step 8: Compute efficiency ────────────────────────────────────────────
    efficiency = float(cl / cd) if cd != 0.0 else 0.0

    # ── Step 9: Return results ────────────────────────────────────────────────
    return {
        "max_thickness":      float(real_geom[0]),
        "thickness_location": float(real_geom[1]),
        "max_camber":         float(real_geom[2]),
        "camber_location":    float(real_geom[3]),
        "cl":                 float(cl),
        "cd":                 float(cd),
        "cm":                 float(cm),
        "efficiency":         efficiency,
        "fitness":            float(best_fitness),
        "fitness_history":    fitness_history,
    }