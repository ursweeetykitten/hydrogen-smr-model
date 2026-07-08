# -*- coding: utf-8 -*-
"""
Step 03. Final material and heat balances for the best operating point.

The script reads selected catalyst masses, fixed pressure drops and the best
T/P/S:C point, then writes final CSV balance tables.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from core_reactor_block import (
    COMPONENTS,
    T_FINAL_COOLING_C,
    atom_balance_table,
    heat_balance_tables,
    results_dir,
    simulate_block,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = results_dir(BASE_DIR / "results")
USE_ETA = True


def load_selected_masses() -> pd.Series:
    path = RESULTS_DIR / "01d_selected_catalyst_masses.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_catalyst_hydraulics_selection.py first")
    return pd.read_csv(path).iloc[0]


def load_best_operation() -> pd.Series:
    path = RESULTS_DIR / "02b_best_operating_point.csv"
    if not path.exists():
        raise FileNotFoundError("Run 02_operating_optimization.py first")
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
    }


def main() -> None:
    selected = load_selected_masses()
    best = load_best_operation()
    fixed_dps = load_fixed_dps()

    res = simulate_block(
        W_smr_loaded_kg=float(selected["W_smr_loaded_kg"]),
        W_hts_loaded_kg=float(selected["W_hts_loaded_kg"]),
        W_lts_loaded_kg=float(selected["W_lts_loaded_kg"]),
        smr_model=str(selected["smr_model"]),
        T_smr_in_C=float(best["T_smr_in_C"]),
        T_smr_out_C=float(best["T_smr_out_C"]),
        T_hts_in_C=float(best["T_hts_in_C"]),
        T_lts_in_C=float(best["T_lts_in_C"]),
        SC=float(best["SC"]),
        P_smr_in_bar=float(best["P_smr_in_bar"]),
        use_eta=USE_ETA,
        dp_mode="fixed",
        fixed_dps=fixed_dps,
        active_tubes=int(selected["active_smr_tubes"]),
    )

    df_mat, df_heat, df_pressure, df_summary = heat_balance_tables(res, final_cooling_T_C=T_FINAL_COOLING_C)
    df_atom = atom_balance_table(res["F0"], res["F_lts"])

    df_mat.to_csv(RESULTS_DIR / "03a_material_balance_by_stream.csv", index=False, encoding="utf-8-sig")
    df_heat.to_csv(RESULTS_DIR / "03b_heat_balance.csv", index=False, encoding="utf-8-sig")
    df_pressure.to_csv(RESULTS_DIR / "03c_pressure_and_hydraulic_balance.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(RESULTS_DIR / "03d_summary.csv", index=False, encoding="utf-8-sig")
    df_atom.to_csv(RESULTS_DIR / "03e_atom_balance_check.csv", index=False, encoding="utf-8-sig")

    print("Step 03 complete.")
    s = df_summary.iloc[0]
    print(f"H2 = {s['H2_net_Nm3_h']:.1f} Nm3/h, H2/CH4 = {s['H2_specific_Nm3_per_Nm3_CH4']:.3f}")
    print(f"X_CH4 = {s['X_CH4_pct']:.2f} %, CO dry = {s['CO_dry_pct']:.3f} %, P_out = {s['P_lts_out_bar']:.2f} bar")
    print(f"SMR total reactor heat = {s['SMR_total_reactor_heat_MW']:.2f} MW")
    print(f"Steam generation = {s['steam_generation_total_MW']:.2f} MW")
    print(f"Mixture heating 380 C to SMR inlet = {s['mixture_heating_380_to_SMR_in_MW']:.2f} MW")
    print(f"Total cooling = {s['cooling_total_MW']:.2f} MW")


if __name__ == "__main__":
    main()
