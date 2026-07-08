# -*- coding: utf-8 -*-
"""
Step 02. Optimization of pressure, temperatures and steam-to-carbon ratio.

Inputs from Step 01:
    - selected catalyst masses
    - fixed pressure drops calculated by Ergun at the base point

Optimized variables:
    - SMR inlet temperature: 750...920 C
    - SMR temperature drop: 0...300 C with T_out >= 740 C
    - HTS inlet temperature: 320...400 C
    - LTS inlet temperature: 190...260 C
    - S/C: 2.0...4.0, rounded to 0.2
    - SMR inlet pressure: minimum hydraulic pressure...40 bar, rounded to 2 bar

The reactors HTS and LTS are adiabatic. The SMR temperature profile is linear.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from core_reactor_block import (
    CAT_HTS,
    CAT_LTS,
    CAT_SMR,
    F_CH4_FEED_NM3_H,
    MAX_INITIAL_PRESSURE_BAR,
    MIN_INITIAL_PRESSURE_BAR,
    MIN_PRODUCT_PRESSURE_BAR,
    PRESSURE_STEP_BAR,
    SC_RANGE,
    TARGET_H2_NM3_H,
    T_HTS_IN_RANGE_C,
    T_HTS_OUT_RANGE_C,
    T_LTS_IN_RANGE_C,
    T_LTS_OUT_RANGE_C,
    T_SMR_DT_RANGE_C,
    T_SMR_IN_RANGE_C,
    T_SMR_OUT_MIN_C,
    ceil_to_step,
    flatten_result,
    heat_balance_tables,
    results_dir,
    simulate_block,
)

# =============================================================================
# EDITABLE SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = results_dir(BASE_DIR / "results")

USE_ETA = True
RANDOM_SEED = 42
MAXITER = 12
POPSIZE = 6
TOL = 1.0e-3

# Discrete step sizes used to round optimizer variables.
T_SMR_IN_STEP_C = 10.0
T_SMR_DT_STEP_C = 10.0
T_HTS_IN_STEP_C = 5.0
T_LTS_IN_STEP_C = 5.0
SC_STEP = 0.2
P_STEP_BAR = 2.0

# Main constraints.
MIN_X_CH4_PCT = 90.0
MAX_CO_DRY_PCT = 1.0
MIN_H2_SPECIFIC = 2.5

# Objective weights. Feasibility penalties dominate; heat and operating costs
# are used only to rank feasible points.
W_X_CH4 = 5000.0
W_CO = 5000.0
W_H2_DEFICIT = 10000.0
W_H2_OVERSHOOT = 0.08
W_PRESSURE = 5000.0
W_TEMPERATURE = 100.0
W_HEAT = 0.02
W_SC = 4.0
W_PRESSURE_SECONDARY = 0.1

# =============================================================================
# FILE READERS
# =============================================================================

def load_selected_masses() -> pd.Series:
    path = RESULTS_DIR / "01d_selected_catalyst_masses.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_catalyst_hydraulics_selection.py first")
    return pd.read_csv(path).iloc[0]


def load_fixed_dps() -> dict:
    path = RESULTS_DIR / "01e_fixed_pressure_drops_for_next_steps.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_catalyst_hydraulics_selection.py first")
    df = pd.read_csv(path)
    return {
        "SMR": float(df.loc[df["stage"] == "SMR", "dP_bar"].iloc[0]),
        "HTS": float(df.loc[df["stage"] == "HTS", "dP_bar"].iloc[0]),
        "LTS": float(df.loc[df["stage"] == "LTS", "dP_bar"].iloc[0]),
        "TOTAL": float(df.loc[df["stage"] == "TOTAL", "dP_bar"].iloc[0]),
    }

# =============================================================================
# OBJECTIVE
# =============================================================================

def round_to_step(value: float, step: float, lo: float, hi: float) -> float:
    y = round((value - lo) / step) * step + lo
    return float(min(max(y, lo), hi))


def pressure_grid_min(fixed_dps: dict) -> float:
    p_req = MIN_PRODUCT_PRESSURE_BAR + fixed_dps["TOTAL"]
    return max(MIN_INITIAL_PRESSURE_BAR, ceil_to_step(p_req, P_STEP_BAR))


def round_variables(x: np.ndarray, p_min: float) -> tuple[float, float, float, float, float, float]:
    T_smr_in = round_to_step(x[0], T_SMR_IN_STEP_C, T_SMR_IN_RANGE_C[0], T_SMR_IN_RANGE_C[1])
    max_dt = min(T_SMR_DT_RANGE_C[1], T_smr_in - T_SMR_OUT_MIN_C)
    max_dt = max(0.0, max_dt)
    dT_smr = round_to_step(x[1], T_SMR_DT_STEP_C, 0.0, max_dt)
    T_hts_in = round_to_step(x[2], T_HTS_IN_STEP_C, T_HTS_IN_RANGE_C[0], T_HTS_IN_RANGE_C[1])
    T_lts_in = round_to_step(x[3], T_LTS_IN_STEP_C, T_LTS_IN_RANGE_C[0], T_LTS_IN_RANGE_C[1])
    SC = round_to_step(x[4], SC_STEP, SC_RANGE[0], SC_RANGE[1])
    P = round_to_step(x[5], P_STEP_BAR, p_min, MAX_INITIAL_PRESSURE_BAR)
    return T_smr_in, dT_smr, T_hts_in, T_lts_in, SC, P


def outside_range_penalty(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return (lo - value) ** 2
    if value > hi:
        return (value - hi) ** 2
    return 0.0


def evaluate_case(selected: pd.Series, fixed_dps: dict, variables: tuple[float, float, float, float, float, float]) -> dict:
    T_smr_in, dT_smr, T_hts_in, T_lts_in, SC, P = variables
    T_smr_out = T_smr_in - dT_smr
    res = simulate_block(
        W_smr_loaded_kg=float(selected["W_smr_loaded_kg"]),
        W_hts_loaded_kg=float(selected["W_hts_loaded_kg"]),
        W_lts_loaded_kg=float(selected["W_lts_loaded_kg"]),
        smr_model=str(selected["smr_model"]),
        T_smr_in_C=T_smr_in,
        T_smr_out_C=T_smr_out,
        T_hts_in_C=T_hts_in,
        T_lts_in_C=T_lts_in,
        SC=SC,
        P_smr_in_bar=P,
        use_eta=USE_ETA,
        dp_mode="fixed",
        fixed_dps=fixed_dps,
        active_tubes=int(selected["active_smr_tubes"]),
    )
    row = flatten_result(res)
    _, heat_df, _, summary_df = heat_balance_tables(res)
    summary = summary_df.iloc[0].to_dict()
    row.update({
        "H2_target_Nm3_h": TARGET_H2_NM3_H,
        "H2_minus_target_Nm3_h": row["H2_net_Nm3_h"] - TARGET_H2_NM3_H,
        "SMR_total_reactor_heat_MW": summary["SMR_total_reactor_heat_MW"],
        "steam_generation_total_MW": summary["steam_generation_total_MW"],
        "mixture_heating_380_to_SMR_in_MW": summary["mixture_heating_380_to_SMR_in_MW"],
        "cooling_total_MW": summary["cooling_total_MW"],
        "thermal_sum_abs_MW": abs(summary["SMR_total_reactor_heat_MW"]) + summary["steam_generation_total_MW"] + summary["mixture_heating_380_to_SMR_in_MW"] + summary["cooling_total_MW"],
    })
    return row


def score_row(row: dict) -> float:
    score = 0.0
    score += W_X_CH4 * max(0.0, MIN_X_CH4_PCT - row["X_CH4_pct"]) ** 2
    score += W_CO * max(0.0, row["CO_dry_pct"] - MAX_CO_DRY_PCT) ** 2
    score += W_H2_DEFICIT * max(0.0, MIN_H2_SPECIFIC - row["H2_specific_Nm3_per_Nm3_CH4"]) ** 2
    score += W_PRESSURE * max(0.0, MIN_PRODUCT_PRESSURE_BAR - row["P_lts_out_bar"]) ** 2
    score += W_TEMPERATURE * max(0.0, T_SMR_OUT_MIN_C - row["T_smr_out_C"]) ** 2
    score += W_TEMPERATURE * outside_range_penalty(row["T_hts_out_C"], *T_HTS_OUT_RANGE_C)
    score += W_TEMPERATURE * outside_range_penalty(row["T_lts_out_C"], *T_LTS_OUT_RANGE_C)

    # Target capacity is treated as a soft target, because CH4 conversion >90 %
    # and 2600 Nm3/h CH4 may naturally give more than 6500 Nm3/h H2 before PSA.
    score += W_H2_OVERSHOOT * max(0.0, row["H2_net_Nm3_h"] - TARGET_H2_NM3_H) ** 2 / TARGET_H2_NM3_H**2

    # Secondary preferences: lower heat load, lower steam ratio and lower pressure.
    score += W_HEAT * row["thermal_sum_abs_MW"]
    score += W_SC * (row["SC"] - SC_RANGE[0]) ** 2
    score += W_PRESSURE_SECONDARY * (row["P_smr_in_bar"] - MIN_INITIAL_PRESSURE_BAR) ** 2
    return float(score)


def optimize() -> tuple[pd.DataFrame, pd.Series]:
    selected = load_selected_masses()
    fixed_dps = load_fixed_dps()
    p_min = pressure_grid_min(fixed_dps)
    print(f"Using fixed pressure drops: {fixed_dps}")
    print(f"Pressure grid: {p_min:.1f}...{MAX_INITIAL_PRESSURE_BAR:.1f} bar, step {P_STEP_BAR:.1f} bar")

    cache: dict[tuple[float, float, float, float, float, float], dict] = {}

    def objective(x: np.ndarray) -> float:
        key = round_variables(x, p_min)
        if key in cache:
            return cache[key]["score"]
        try:
            row = evaluate_case(selected, fixed_dps, key)
            row["score"] = score_row(row)
            row["success"] = True
        except Exception as exc:
            T_smr_in, dT_smr, T_hts_in, T_lts_in, SC, P = key
            row = {
                "T_smr_in_C": T_smr_in,
                "smr_delta_T_C": dT_smr,
                "T_smr_out_C": T_smr_in - dT_smr,
                "T_hts_in_C": T_hts_in,
                "T_lts_in_C": T_lts_in,
                "SC": SC,
                "P_smr_in_bar": P,
                "score": 1.0e12,
                "success": False,
                "error": repr(exc),
            }
        cache[key] = row
        return row["score"]

    bounds = [
        T_SMR_IN_RANGE_C,
        T_SMR_DT_RANGE_C,
        T_HTS_IN_RANGE_C,
        T_LTS_IN_RANGE_C,
        SC_RANGE,
        (p_min, MAX_INITIAL_PRESSURE_BAR),
    ]
    differential_evolution(
        objective,
        bounds=bounds,
        seed=RANDOM_SEED,
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=TOL,
        polish=True,
        workers=1,
        updating="immediate",
        disp=False,
    )
    df = pd.DataFrame(cache.values()).sort_values("score")
    feasible = df[
        (df["X_CH4_pct"] >= MIN_X_CH4_PCT)
        & (df["CO_dry_pct"] <= MAX_CO_DRY_PCT)
        & (df["H2_specific_Nm3_per_Nm3_CH4"] >= MIN_H2_SPECIFIC)
        & (df["P_lts_out_bar"] >= MIN_PRODUCT_PRESSURE_BAR)
        & (df["T_smr_out_C"] >= T_SMR_OUT_MIN_C)
        & (df["T_hts_out_C"] >= T_HTS_OUT_RANGE_C[0])
        & (df["T_hts_out_C"] <= T_HTS_OUT_RANGE_C[1])
        & (df["T_lts_out_C"] >= T_LTS_OUT_RANGE_C[0])
        & (df["T_lts_out_C"] <= T_LTS_OUT_RANGE_C[1])
    ].copy()
    if not feasible.empty:
        best = feasible.sort_values("score").iloc[0]
    else:
        best = df.iloc[0]
    return df, best


def main() -> None:
    df, best = optimize()
    df.to_csv(RESULTS_DIR / "02a_operating_optimization_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best.to_dict()]).to_csv(RESULTS_DIR / "02b_best_operating_point.csv", index=False, encoding="utf-8-sig")
    print("Step 02 complete.")
    print(f"Best point: T_SMR={best['T_smr_in_C']:.0f}/{best['T_smr_out_C']:.0f} C, HTS={best['T_hts_in_C']:.0f} C, LTS={best['T_lts_in_C']:.0f} C, S/C={best['SC']:.1f}, P={best['P_smr_in_bar']:.0f} bar")
    print(f"X_CH4={best['X_CH4_pct']:.2f} %, CO dry={best['CO_dry_pct']:.3f} %, H2={best['H2_net_Nm3_h']:.1f} Nm3/h, P_out={best['P_lts_out_bar']:.2f} bar")
    print(f"Thermal sum abs={best['thermal_sum_abs_MW']:.2f} MW, score={best['score']:.3g}")


if __name__ == "__main__":
    main()
