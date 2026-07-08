# -*- coding: utf-8 -*-
"""
Step 01. Catalyst mass selection and primary hydraulic calculation.

The script performs two linked studies at the base operating point:
    1) Kinetic mass sweep: minimum catalyst loading that satisfies
       X(CH4) >= 90 %, dry CO <= 1 %, H2/CH4 > 2.5.
    2) Hydraulic check by the Ergun equation for SMR, HTS and LTS.

The selected pressure drops are saved and reused by Step 02 and Step 03.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from core_reactor_block import (
    BASE_HTS_T_IN_C,
    BASE_LTS_T_IN_C,
    BASE_P_IN_BAR,
    BASE_SC,
    BASE_SMR_T_IN_C,
    BASE_SMR_T_OUT_C,
    CAT_HTS,
    CAT_LTS,
    CAT_SMR,
    COMPONENTS,
    DP_COOLER_BEFORE_HTS_BAR,
    DP_COOLER_BEFORE_LTS_BAR,
    F_CH4_FEED_NM3_H,
    MAX_INITIAL_PRESSURE_BAR,
    MIN_INITIAL_PRESSURE_BAR,
    MIN_PRODUCT_PRESSURE_BAR,
    PRESSURE_STEP_BAR,
    TARGET_H2_NM3_H,
    T_HTS_OUT_RANGE_C,
    T_LTS_OUT_RANGE_C,
    MAX_SMR_DP_BAR,
    MAX_SMR_U_M_S,
    MIN_SMR_U_M_S,
    REACTOR_GEOM,
    dry_co_percent,
    ergun_dp_dz_pa_m,
    find_min_pressure_for_case,
    flatten_result,
    make_feed,
    nm3_h_from_kmol_h,
    results_dir,
    run_hts_reactor,
    run_lts_reactor,
    run_smr_reactor,
    select_active_smr_tubes,
    simulate_block,
)

# =============================================================================
# EDITABLE SETTINGS
# =============================================================================

RESULTS_DIR = results_dir(Path(__file__).resolve().parent / "results")

# Default model is Hou-Hughes. Add "Xu-Froment" to this list if you want to
# compare both SMR kinetic descriptions.
SMR_MODELS = ["Hou-Hughes"]

USE_ETA = True

SMR_W_NOMINAL_RANGE_KG = np.arange(50.0, 2000.0 + 50.0, 50.0)
HTS_W_NOMINAL_RANGE_KG = np.arange(50.0, 1000.0 + 50.0, 50.0)
LTS_W_NOMINAL_RANGE_KG = np.arange(50.0, 1000.0 + 50.0, 50.0)

# Base point requested for the first calculation.
BASE_TUPLE = {
    "SC": BASE_SC,
    "P_smr_in_bar": BASE_P_IN_BAR,
    "T_smr_in_C": BASE_SMR_T_IN_C,
    "T_smr_out_C": BASE_SMR_T_OUT_C,
    "T_hts_in_C": BASE_HTS_T_IN_C,
    "T_lts_in_C": BASE_LTS_T_IN_C,
}

# Selection limits.
MIN_X_CH4_PCT = 90.0
MAX_CO_DRY_PCT = 1.0
MIN_H2_SPECIFIC = 2.5

# Use a faster coarse discretization for the mass sweep. The selected candidate
# is re-run with the default discretization in final_balance.py.
MASS_SWEEP_N_SMR_LAYERS = 100
MASS_SWEEP_N_HTS_LAYERS = 60
MASS_SWEEP_N_LTS_LAYERS = 70

# =============================================================================
# HYDRAULIC BASE CHECK
# =============================================================================

def primary_smr_tube_hydraulics() -> pd.DataFrame:
    F0 = make_feed(SC=BASE_SC)
    T_avg_C = 0.5 * (BASE_SMR_T_IN_C + BASE_SMR_T_OUT_C)
    A_tube = 0.25 * np.pi * REACTOR_GEOM["smr_d_inner_m"] ** 2
    rows = []
    for mode, tubes in [
        ("all_400_installed_tubes", int(REACTOR_GEOM["smr_tubes_installed"])),
        ("auto_selected_active_tubes", select_active_smr_tubes(F0, T_avg_C, BASE_P_IN_BAR)),
    ]:
        F_tube = {c: F0[c] / tubes for c in COMPONENTS}
        dpdz, u = ergun_dp_dz_pa_m(F_tube, T_avg_C + 273.15, BASE_P_IN_BAR, A_tube, CAT_SMR)
        rows.append({
            "mode": mode,
            "active_tubes": tubes,
            "SC": BASE_SC,
            "P_bar": BASE_P_IN_BAR,
            "T_avg_C": T_avg_C,
            "u_s_m_s": u,
            "dP_for_full_12m_bar": dpdz * REACTOR_GEOM["smr_tube_len_m"] / 1.0e5,
            "velocity_ok_1_5_m_s": bool(MIN_SMR_U_M_S <= u <= MAX_SMR_U_M_S),
        })
    return pd.DataFrame(rows)

# =============================================================================
# MASS SWEEP
# =============================================================================

def add_selection_score(row: dict) -> dict:
    penalties = 0.0
    penalties += 1000.0 * max(0.0, MIN_X_CH4_PCT - row.get("X_CH4_pct", 0.0)) ** 2
    penalties += 1000.0 * max(0.0, row.get("CO_dry_pct", 1.0e9) - MAX_CO_DRY_PCT) ** 2
    penalties += 5000.0 * max(0.0, MIN_H2_SPECIFIC - row.get("H2_specific_Nm3_per_Nm3_CH4", 0.0)) ** 2
    penalties += 1000.0 * max(0.0, MIN_PRODUCT_PRESSURE_BAR - row.get("P_lts_out_bar", 0.0)) ** 2
    penalties += 500.0 * max(0.0, row.get("SMR_dP_bar", 0.0) - MAX_SMR_DP_BAR) ** 2
    if not row.get("SMR_velocity_ok", False):
        u = row.get("SMR_u_avg_m_s", 0.0)
        if u < MIN_SMR_U_M_S:
            penalties += 500.0 * (MIN_SMR_U_M_S - u) ** 2
        if u > MAX_SMR_U_M_S:
            penalties += 500.0 * (u - MAX_SMR_U_M_S) ** 2
    if not row.get("SMR_bed_length_ok", False):
        penalties += 10000.0
    row["strict_feasible"] = bool(
        row.get("X_CH4_pct", 0.0) >= MIN_X_CH4_PCT
        and row.get("CO_dry_pct", 1.0e9) <= MAX_CO_DRY_PCT
        and row.get("H2_specific_Nm3_per_Nm3_CH4", 0.0) >= MIN_H2_SPECIFIC
        and row.get("P_lts_out_bar", 0.0) >= MIN_PRODUCT_PRESSURE_BAR
        and row.get("SMR_dP_bar", 1.0e9) <= MAX_SMR_DP_BAR
        and row.get("SMR_velocity_ok", False)
        and row.get("SMR_bed_length_ok", False)
    )
    row["word_temperature_window_ok"] = bool(
        T_HTS_OUT_RANGE_C[0] <= row.get("T_hts_out_C", -1.0e9) <= T_HTS_OUT_RANGE_C[1]
        and T_LTS_OUT_RANGE_C[0] <= row.get("T_lts_out_C", -1.0e9) <= T_LTS_OUT_RANGE_C[1]
    )
    row["strict_feasible_with_word_temperature"] = bool(row["strict_feasible"] and row["word_temperature_window_ok"])
    if not row["word_temperature_window_ok"]:
        penalties += 100.0 * max(0.0, T_HTS_OUT_RANGE_C[0] - row.get("T_hts_out_C", 0.0)) ** 2
        penalties += 100.0 * max(0.0, row.get("T_hts_out_C", 0.0) - T_HTS_OUT_RANGE_C[1]) ** 2
        penalties += 100.0 * max(0.0, T_LTS_OUT_RANGE_C[0] - row.get("T_lts_out_C", 0.0)) ** 2
        penalties += 100.0 * max(0.0, row.get("T_lts_out_C", 0.0) - T_LTS_OUT_RANGE_C[1]) ** 2
    row["selection_score"] = penalties + 0.001 * row.get("W_total_loaded_kg", 0.0)
    return row


def build_row_from_reactors(
    smr_model: str,
    W_smr_nom: float,
    W_hts_nom: float,
    W_lts_nom: float,
    W_smr_loaded: float,
    W_hts_loaded: float,
    W_lts_loaded: float,
    active_tubes: int,
    smr,
    hts,
    lts,
    P_smr_in_bar: float,
) -> dict:
    F0 = make_feed(SC=BASE_SC)
    H2_net_kmol_h = lts.F_out["H2"] - F0["H2"]
    H2_net_Nm3_h = nm3_h_from_kmol_h(H2_net_kmol_h)
    row = {
        "smr_model": smr_model,
        "W_smr_nominal_kg": W_smr_nom,
        "W_hts_nominal_kg": W_hts_nom,
        "W_lts_nominal_kg": W_lts_nom,
        "K_res_SMR": CAT_SMR["K_res"],
        "K_res_HTS": CAT_HTS["K_res"],
        "K_res_LTS": CAT_LTS["K_res"],
        "eta_eff_SMR": CAT_SMR["eta_eff"],
        "eta_eff_HTS": CAT_HTS["eta_eff"],
        "eta_eff_LTS": CAT_LTS["eta_eff"],
        "W_smr_loaded_kg": W_smr_loaded,
        "W_hts_loaded_kg": W_hts_loaded,
        "W_lts_loaded_kg": W_lts_loaded,
        "W_total_nominal_kg": W_smr_nom + W_hts_nom + W_lts_nom,
        "W_total_loaded_kg": W_smr_loaded + W_hts_loaded + W_lts_loaded,
        "active_smr_tubes": active_tubes,
        "T_smr_in_C": BASE_SMR_T_IN_C,
        "T_smr_out_C": BASE_SMR_T_OUT_C,
        "T_hts_in_C": BASE_HTS_T_IN_C,
        "T_hts_out_C": hts.T_out_K - 273.15,
        "T_lts_in_C": BASE_LTS_T_IN_C,
        "T_lts_out_C": lts.T_out_K - 273.15,
        "SC": BASE_SC,
        "P_smr_in_bar": P_smr_in_bar,
        "P_smr_out_bar": smr.P_out_bar,
        "P_hts_in_bar": smr.P_out_bar - DP_COOLER_BEFORE_HTS_BAR,
        "P_hts_out_bar": hts.P_out_bar,
        "P_lts_in_bar": hts.P_out_bar - DP_COOLER_BEFORE_LTS_BAR,
        "P_lts_out_bar": lts.P_out_bar,
        "P_required_for_16bar_product": MIN_PRODUCT_PRESSURE_BAR + (P_smr_in_bar - lts.P_out_bar),
        "pressure_increased_from_25bar": bool(P_smr_in_bar > BASE_P_IN_BAR + 1.0e-9),
        "X_CH4_pct": (F0["CH4"] - lts.F_out["CH4"]) / F0["CH4"] * 100.0,
        "X_CO_HTS_pct": (smr.F_out["CO"] - hts.F_out["CO"]) / max(smr.F_out["CO"], 1.0e-12) * 100.0,
        "X_CO_LTS_pct": (hts.F_out["CO"] - lts.F_out["CO"]) / max(hts.F_out["CO"], 1.0e-12) * 100.0,
        "CO_dry_pct": dry_co_percent(lts.F_out),
        "H2_net_Nm3_h": H2_net_Nm3_h,
        "H2_specific_Nm3_per_Nm3_CH4": H2_net_Nm3_h / F_CH4_FEED_NM3_H,
        "Q_SMR_reaction_MW": smr.Q_reaction_kJ_h / 3.6e6,
        "Q_HTS_reaction_MW": hts.Q_reaction_kJ_h / 3.6e6,
        "Q_LTS_reaction_MW": lts.Q_reaction_kJ_h / 3.6e6,
        "SMR_dP_bar": smr.geom["dP_bar"],
        "HTS_dP_bar": hts.geom["dP_bar"],
        "LTS_dP_bar": lts.geom["dP_bar"],
        "SMR_u_avg_m_s": smr.geom["u_s_avg_m_s"],
        "HTS_u_avg_m_s": hts.geom["u_s_avg_m_s"],
        "LTS_u_avg_m_s": lts.geom["u_s_avg_m_s"],
        "SMR_bed_length_m": smr.geom["bed_length_m"],
        "HTS_bed_length_m": hts.geom["bed_length_m"],
        "LTS_bed_length_m": lts.geom["bed_length_m"],
        "HTS_d_vessel_m": hts.geom["d_vessel_m"],
        "LTS_d_vessel_m": lts.geom["d_vessel_m"],
        "HTS_velocity_ok": hts.geom.get("velocity_ok", None),
        "LTS_velocity_ok": lts.geom.get("velocity_ok", None),
        "SMR_velocity_ok": smr.geom["velocity_ok"],
        "SMR_dP_ok": smr.geom["dP_ok"],
        "SMR_bed_length_ok": smr.geom["bed_length_ok"],
    }
    return add_selection_score(row)


def run_mass_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    F0 = make_feed(SC=BASE_SC)
    active_tubes = select_active_smr_tubes(F0, 0.5 * (BASE_SMR_T_IN_C + BASE_SMR_T_OUT_C), BASE_P_IN_BAR)
    smr_rows = []
    chain_rows = []

    for smr_model in SMR_MODELS:
        print(f"SMR model: {smr_model}")
        valid_smr_cache = []
        for W_smr_nom in SMR_W_NOMINAL_RANGE_KG:
            W_smr_loaded = W_smr_nom * CAT_SMR["K_res"]
            smr = run_smr_reactor(
                F0, W_smr_loaded, smr_model, BASE_SMR_T_IN_C, BASE_SMR_T_OUT_C,
                BASE_P_IN_BAR, active_tubes=active_tubes, use_eta=USE_ETA,
                dp_mode="ergun", n_layers=MASS_SWEEP_N_SMR_LAYERS,
            )
            X_smr = (F0["CH4"] - smr.F_out["CH4"]) / F0["CH4"] * 100.0
            smr_row = {
                "smr_model": smr_model,
                "W_smr_nominal_kg": W_smr_nom,
                "W_smr_loaded_kg": W_smr_loaded,
                "X_CH4_after_SMR_pct": X_smr,
                "CO_dry_after_SMR_pct": dry_co_percent(smr.F_out),
                "H2_after_SMR_Nm3_h": nm3_h_from_kmol_h(smr.F_out["H2"] - F0["H2"]),
                "SMR_dP_bar": smr.geom["dP_bar"],
                "SMR_u_avg_m_s": smr.geom["u_s_avg_m_s"],
                "SMR_bed_length_m": smr.geom["bed_length_m"],
                "SMR_velocity_ok": smr.geom["velocity_ok"],
                "SMR_dP_ok": smr.geom["dP_ok"],
                "SMR_bed_length_ok": smr.geom["bed_length_ok"],
                "passes_SMR_prescreen_X_CH4_90": bool(X_smr >= MIN_X_CH4_PCT and smr.geom["velocity_ok"] and smr.geom["dP_ok"] and smr.geom["bed_length_ok"]),
            }
            smr_rows.append(smr_row)
            if smr_row["passes_SMR_prescreen_X_CH4_90"]:
                valid_smr_cache.append((W_smr_nom, W_smr_loaded, smr))

        if not valid_smr_cache:
            # Still test the best SMR candidate if strict 90 % is unreachable.
            best_smr_row = sorted(smr_rows, key=lambda r: r["X_CH4_after_SMR_pct"], reverse=True)[0]
            W_smr_nom = best_smr_row["W_smr_nominal_kg"]
            W_smr_loaded = best_smr_row["W_smr_loaded_kg"]
            smr = run_smr_reactor(
                F0, W_smr_loaded, smr_model, BASE_SMR_T_IN_C, BASE_SMR_T_OUT_C,
                BASE_P_IN_BAR, active_tubes=active_tubes, use_eta=USE_ETA,
                dp_mode="ergun", n_layers=MASS_SWEEP_N_SMR_LAYERS,
            )
            valid_smr_cache.append((W_smr_nom, W_smr_loaded, smr))

        print(f"  SMR candidates after prescreen: {len(valid_smr_cache)}")
        for idx, (W_smr_nom, W_smr_loaded, smr) in enumerate(valid_smr_cache, start=1):
            if idx % 5 == 0 or idx == 1:
                print(f"  chain sweep for SMR candidate {idx}/{len(valid_smr_cache)}")
            P_hts_in = smr.P_out_bar - DP_COOLER_BEFORE_HTS_BAR
            for W_hts_nom in HTS_W_NOMINAL_RANGE_KG:
                W_hts_loaded = W_hts_nom * CAT_HTS["K_res"]
                hts = run_hts_reactor(
                    smr.F_out, W_hts_loaded, BASE_HTS_T_IN_C, P_hts_in,
                    use_eta=USE_ETA, dp_mode="ergun", n_layers=MASS_SWEEP_N_HTS_LAYERS,
                )
                P_lts_in = hts.P_out_bar - DP_COOLER_BEFORE_LTS_BAR
                for W_lts_nom in LTS_W_NOMINAL_RANGE_KG:
                    W_lts_loaded = W_lts_nom * CAT_LTS["K_res"]
                    lts = run_lts_reactor(
                        hts.F_out, W_lts_loaded, BASE_LTS_T_IN_C, P_lts_in,
                        use_eta=USE_ETA, dp_mode="ergun", n_layers=MASS_SWEEP_N_LTS_LAYERS,
                    )
                    row = build_row_from_reactors(
                        smr_model, W_smr_nom, W_hts_nom, W_lts_nom,
                        W_smr_loaded, W_hts_loaded, W_lts_loaded,
                        active_tubes, smr, hts, lts, BASE_P_IN_BAR,
                    )
                    # If the base pressure is insufficient, re-run only this case with a higher pressure.
                    if row["P_lts_out_bar"] < MIN_PRODUCT_PRESSURE_BAR:
                        P_sel, res_full = find_min_pressure_for_case(
                            W_smr_loaded, W_hts_loaded, W_lts_loaded, smr_model,
                            BASE_SMR_T_IN_C, BASE_SMR_T_OUT_C, BASE_HTS_T_IN_C, BASE_LTS_T_IN_C,
                            BASE_SC, active_tubes, use_eta=USE_ETA, dp_mode="ergun",
                            P_start_bar=MIN_INITIAL_PRESSURE_BAR, P_stop_bar=MAX_INITIAL_PRESSURE_BAR,
                            P_step_bar=PRESSURE_STEP_BAR,
                        )
                        row = flatten_result(res_full)
                        row.update({
                            "W_smr_nominal_kg": W_smr_nom,
                            "W_hts_nominal_kg": W_hts_nom,
                            "W_lts_nominal_kg": W_lts_nom,
                            "K_res_SMR": CAT_SMR["K_res"],
                            "K_res_HTS": CAT_HTS["K_res"],
                            "K_res_LTS": CAT_LTS["K_res"],
                            "eta_eff_SMR": CAT_SMR["eta_eff"],
                            "eta_eff_HTS": CAT_HTS["eta_eff"],
                            "eta_eff_LTS": CAT_LTS["eta_eff"],
                            "W_total_nominal_kg": W_smr_nom + W_hts_nom + W_lts_nom,
                            "W_total_loaded_kg": W_smr_loaded + W_hts_loaded + W_lts_loaded,
                            "P_selected_bar": P_sel,
                            "pressure_increased_from_25bar": bool(P_sel > BASE_P_IN_BAR + 1.0e-9),
                        })
                        row = add_selection_score(row)
                    chain_rows.append(row)

    df_smr = pd.DataFrame(smr_rows)
    df_chain = pd.DataFrame(chain_rows)
    feasible_temp = df_chain[df_chain["strict_feasible_with_word_temperature"] == True].copy()
    feasible = df_chain[df_chain["strict_feasible"] == True].copy()
    if not feasible_temp.empty:
        selected = feasible_temp.sort_values(["W_total_loaded_kg", "W_smr_loaded_kg", "selection_score"]).iloc[0]
    elif not feasible.empty:
        selected = feasible.sort_values(["W_total_loaded_kg", "W_smr_loaded_kg", "selection_score"]).iloc[0]
    else:
        selected = df_chain.sort_values("selection_score").iloc[0]
    return df_smr, df_chain, selected


def fixed_pressure_drop_table(selected: pd.Series) -> pd.DataFrame:
    rows = [
        {"stage": "SMR", "dP_bar": selected["SMR_dP_bar"], "source": "Ergun at selected catalyst masses and base point"},
        {"stage": "cooler_before_HTS", "dP_bar": DP_COOLER_BEFORE_HTS_BAR, "source": "user specified"},
        {"stage": "HTS", "dP_bar": selected["HTS_dP_bar"], "source": "Ergun at selected catalyst masses and base point"},
        {"stage": "cooler_before_LTS", "dP_bar": DP_COOLER_BEFORE_LTS_BAR, "source": "user specified"},
        {"stage": "LTS", "dP_bar": selected["LTS_dP_bar"], "source": "Ergun at selected catalyst masses and base point"},
    ]
    total = sum(r["dP_bar"] for r in rows)
    rows.append({"stage": "TOTAL", "dP_bar": total, "source": "reactors + two coolers"})
    rows.append({
        "stage": "minimum_SMR_inlet_pressure_for_16bar_after_LTS",
        "dP_bar": MIN_PRODUCT_PRESSURE_BAR + total,
        "source": "16 bar product pressure + total pressure drop",
    })
    # Selected adiabatic vessel diameters (sized for 0.3-1.0 m/s gas velocity).
    if "HTS_d_vessel_m" in selected.index:
        rows.append({
            "stage": "HTS_vessel_diameter_m",
            "dP_bar": float(selected["HTS_d_vessel_m"]),
            "source": "diameter sized for superficial velocity 0.3-1.0 m/s",
        })
    if "LTS_d_vessel_m" in selected.index:
        rows.append({
            "stage": "LTS_vessel_diameter_m",
            "dP_bar": float(selected["LTS_d_vessel_m"]),
            "source": "diameter sized for superficial velocity 0.3-1.0 m/s",
        })
    return pd.DataFrame(rows)


def main() -> None:
    hyd0 = primary_smr_tube_hydraulics()
    hyd0.to_csv(RESULTS_DIR / "01a_primary_smr_tube_hydraulics.csv", index=False, encoding="utf-8-sig")

    df_smr, df_chain, selected = run_mass_sweep()
    df_smr.to_csv(RESULTS_DIR / "01b_smr_mass_prescreen.csv", index=False, encoding="utf-8-sig")
    df_chain.to_csv(RESULTS_DIR / "01c_catalyst_chain_mass_sweep.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([selected.to_dict()]).to_csv(RESULTS_DIR / "01d_selected_catalyst_masses.csv", index=False, encoding="utf-8-sig")

    dp_table = fixed_pressure_drop_table(selected)
    dp_table.to_csv(RESULTS_DIR / "01e_fixed_pressure_drops_for_next_steps.csv", index=False, encoding="utf-8-sig")

    print("Step 01 complete.")
    print(f"Results directory: {RESULTS_DIR}")
    print("Selected masses:")
    print(f"  SMR nominal/loaded: {selected['W_smr_nominal_kg']:.1f} / {selected['W_smr_loaded_kg']:.1f} kg")
    print(f"  HTS nominal/loaded: {selected['W_hts_nominal_kg']:.1f} / {selected['W_hts_loaded_kg']:.1f} kg")
    print(f"  LTS nominal/loaded: {selected['W_lts_nominal_kg']:.1f} / {selected['W_lts_loaded_kg']:.1f} kg")
    print(f"  P in/out: {selected['P_smr_in_bar']:.2f} / {selected['P_lts_out_bar']:.2f} bar")
    if "HTS_d_vessel_m" in selected.index:
        print(f"  HTS/LTS vessel diameter (sized for 0.3-1.0 m/s): "
              f"{float(selected['HTS_d_vessel_m']):.2f} / {float(selected['LTS_d_vessel_m']):.2f} m")
    print(f"  X_CH4={selected['X_CH4_pct']:.2f} %, CO dry={selected['CO_dry_pct']:.3f} %, H2/CH4={selected['H2_specific_Nm3_per_Nm3_CH4']:.3f}")
    print(f"  Strict feasible: {bool(selected['strict_feasible'])}")


if __name__ == "__main__":
    main()
