# -*- coding: utf-8 -*-
"""
04_separator_psa_simple.py
==========================

Расчет блока после реакторов SMR-HTS-LTS:
  1) охлаждение газа после LTS до 60 C;
  2) газожидкостный сепаратор с конденсацией воды;
  3) PSA-блок через заданную/расчетную степень извлечения H2;
  4) товарный водород 6500 Nm3/h и хвостовой газ PSA.

Файл нужно запускать после 01, 02 и 03, потому что он берет поток LTS_out
из results/03a_material_balance_by_stream.csv.

Создаваемые CSV:
  results/04a_separator_balance.csv
  results/04b_psa_balance.csv
  results/04c_final_product_and_tail_gas_summary.csv

Модель PSA здесь НЕ является динамической адсорбционной моделью.
Это инженерная балансовая модель через степень извлечения H2 и малый проскок
примесей в продукт.
"""

from pathlib import Path
import math
import pandas as pd

# =====================================================================
# 1. НАСТРОЙКИ, КОТОРЫЕ МОЖНО МЕНЯТЬ
# =====================================================================

RESULTS_DIR = Path("results")
INPUT_MATERIAL_BALANCE = RESULTS_DIR / "03a_material_balance_by_stream.csv"

# Условия после конечного холодильника и сепаратора
SEPARATOR_T_C = 60.0
FINAL_COOLER_DP_BAR = 0.50          # перепад давления на холодильнике после LTS, бар

# Целевой товарный H2 после PSA
TARGET_H2_PRODUCT_NM3_H = 6500.0

# Чистота товарного H2. 0.999 = 99.9 мол.%
# Если сумма проскоков примесей даст более грязный продукт, код автоматически
# масштабирует примеси так, чтобы чистота была не ниже PRODUCT_H2_PURITY_MIN.
PRODUCT_H2_PURITY_MIN = 0.999

# Если PSA_H2_RECOVERY_FIXED = None, степень извлечения считается автоматически:
# R_H2 = TARGET_H2_PRODUCT_NM3_H / H2_to_PSA_Nm3_h.
# Если нужно задать вручную, например 0.70, напишите PSA_H2_RECOVERY_FIXED = 0.70.
PSA_H2_RECOVERY_FIXED = None

# Малый проскок примесей в товарный H2, доля от потока компонента на входе PSA.
# Остальная часть уходит в хвостовой газ PSA.
# Эти значения можно менять для чувствительного анализа.
PSA_PRODUCT_SLIP = {
    "CH4": 0.0010,
    "CO":  0.0010,
    "CO2": 0.0005,
    "N2":  0.0050,
    "H2O": 0.0000,
}

# Давления потоков после PSA, приближенно
PSA_PRODUCT_DP_BAR = 0.20           # потеря давления на продуктовой линии PSA, бар
PSA_TAIL_GAS_PRESSURE_BAR = 1.30    # хвостовой газ после сброса давления, бар

# Нижняя теплота сгорания горючих компонентов, МДж/Nm3
# Используется только для оценки топливного потенциала хвостового газа PSA.
LHV_MJ_PER_NM3 = {
    "H2": 10.8,
    "CH4": 35.8,
    "CO": 12.63,
}

# =====================================================================
# 2. КОНСТАНТЫ
# =====================================================================

VM_N = 22.414  # Nm3/kmol
COMPONENTS = ["CH4", "H2O", "CO", "CO2", "H2", "N2"]
MW = {
    "CH4": 16.043,
    "H2O": 18.015,
    "CO": 28.010,
    "CO2": 44.010,
    "H2": 2.016,
    "N2": 28.014,
}

# =====================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================


def psat_water_bar_antoine(T_C: float) -> float:
    """
    Давление насыщенного пара воды, bar.
    Antoine: log10(P_mmHg) = A - B / (C + T_C)
    Диапазон подходит для воды примерно 1-100 C.
    """
    A = 8.07131
    B = 1730.63
    C = 233.426
    p_mmhg = 10.0 ** (A - B / (C + T_C))
    return p_mmhg * 0.00133322368


def load_stream_from_csv(df: pd.DataFrame, stream_name: str):
    """Загружает компонентный поток из 03a_material_balance_by_stream.csv."""
    sub = df[(df["stream"] == stream_name) & (df["component"].isin(COMPONENTS))]
    if sub.empty:
        available = ", ".join(sorted(df["stream"].dropna().unique()))
        raise ValueError(
            f"Stream '{stream_name}' not found in {INPUT_MATERIAL_BALANCE}. "
            f"Available streams: {available}"
        )

    F = {comp: 0.0 for comp in COMPONENTS}
    for _, row in sub.iterrows():
        F[row["component"]] = float(row["kmol_h"])

    T_C = float(sub["T_C"].iloc[0])
    P_bar = float(sub["P_bar"].iloc[0])
    return F, T_C, P_bar


def stream_total(F):
    return sum(F.values())


def dry_total(F):
    return sum(F[c] for c in COMPONENTS if c != "H2O")


def stream_rows(stream_name: str, F: dict, T_C: float, P_bar: float):
    """Формирует строки CSV по компонентам + TOTAL."""
    total = stream_total(F)
    dry = dry_total(F)
    rows = []

    for comp in COMPONENTS:
        kmol_h = F.get(comp, 0.0)
        rows.append({
            "stream": stream_name,
            "T_C": T_C,
            "P_bar": P_bar,
            "component": comp,
            "kmol_h": kmol_h,
            "kg_h": kmol_h * MW[comp],
            "Nm3_h": kmol_h * VM_N,
            "wet_mol_frac": kmol_h / total if total > 0 else 0.0,
            "dry_mol_frac": (kmol_h / dry if dry > 0 and comp != "H2O" else None),
        })

    rows.append({
        "stream": stream_name,
        "T_C": T_C,
        "P_bar": P_bar,
        "component": "TOTAL",
        "kmol_h": total,
        "kg_h": sum(F[c] * MW[c] for c in COMPONENTS),
        "Nm3_h": total * VM_N,
        "wet_mol_frac": 1.0 if total > 0 else 0.0,
        "dry_mol_frac": 1.0 if dry > 0 else None,
    })
    return rows


def lhv_power_MW(F: dict) -> float:
    """Топливный потенциал газа по LHV, MW."""
    mj_per_h = 0.0
    for comp, lhv in LHV_MJ_PER_NM3.items():
        mj_per_h += F.get(comp, 0.0) * VM_N * lhv
    return mj_per_h / 3600.0

# =====================================================================
# 4. СЕПАРАТОР
# =====================================================================


def calculate_separator(F_lts: dict, T_lts_C: float, P_lts_bar: float):
    """
    Охлаждение после LTS до 60 C с потерей давления FINAL_COOLER_DP_BAR.
    Вода в газе после сепаратора определяется по y_H2O = Psat / P.
    Остальная вода считается конденсатом.
    """
    P_sep_bar = P_lts_bar - FINAL_COOLER_DP_BAR
    if P_sep_bar <= 0:
        raise ValueError("Separator pressure became non-positive. Check FINAL_COOLER_DP_BAR.")

    p_sat = psat_water_bar_antoine(SEPARATOR_T_C)
    y_h2o_sat = p_sat / P_sep_bar
    y_h2o_sat = min(max(y_h2o_sat, 0.0), 0.95)

    non_water_kmol_h = dry_total(F_lts)
    h2o_vapor_allowed = non_water_kmol_h * y_h2o_sat / (1.0 - y_h2o_sat)
    h2o_vapor = min(F_lts["H2O"], h2o_vapor_allowed)
    h2o_condensed = max(F_lts["H2O"] - h2o_vapor, 0.0)

    F_gas = dict(F_lts)
    F_gas["H2O"] = h2o_vapor

    F_cond = {comp: 0.0 for comp in COMPONENTS}
    F_cond["H2O"] = h2o_condensed

    info = {
        "P_sep_bar": P_sep_bar,
        "T_sep_C": SEPARATOR_T_C,
        "P_sat_H2O_bar": p_sat,
        "y_H2O_saturation": y_h2o_sat,
        "H2O_condensed_kmol_h": h2o_condensed,
        "H2O_condensed_kg_h": h2o_condensed * MW["H2O"],
        "H2O_condensed_Nm3_h_equiv": h2o_condensed * VM_N,
    }
    return F_gas, F_cond, info

# =====================================================================
# 5. PSA ЧЕРЕЗ СТЕПЕНЬ ИЗВЛЕЧЕНИЯ
# =====================================================================


def calculate_psa(F_psa_feed: dict, P_psa_feed_bar: float):
    """
    PSA-блок через степень извлечения H2 и малый проскок примесей.
    По умолчанию R_H2 подбирается так, чтобы получить TARGET_H2_PRODUCT_NM3_H.
    """
    h2_feed_nm3_h = F_psa_feed["H2"] * VM_N
    if h2_feed_nm3_h <= 0:
        raise ValueError("No H2 in PSA feed.")

    if PSA_H2_RECOVERY_FIXED is None:
        h2_recovery = TARGET_H2_PRODUCT_NM3_H / h2_feed_nm3_h
    else:
        h2_recovery = PSA_H2_RECOVERY_FIXED

    h2_recovery = min(max(h2_recovery, 0.0), 0.999999)
    h2_product_kmol_h = F_psa_feed["H2"] * h2_recovery

    product = {comp: 0.0 for comp in COMPONENTS}
    product["H2"] = h2_product_kmol_h

    # Предварительный расчет проскока примесей в продукт
    for comp, slip in PSA_PRODUCT_SLIP.items():
        if comp == "H2":
            continue
        product[comp] = F_psa_feed.get(comp, 0.0) * slip

    # Проверка чистоты H2: если примесей слишком много, масштабируем их вниз.
    impurity_kmol_h = sum(product[c] for c in COMPONENTS if c != "H2")
    max_impurity_kmol_h = h2_product_kmol_h * (1.0 - PRODUCT_H2_PURITY_MIN) / PRODUCT_H2_PURITY_MIN
    impurity_scale = 1.0
    if impurity_kmol_h > max_impurity_kmol_h and impurity_kmol_h > 0:
        impurity_scale = max_impurity_kmol_h / impurity_kmol_h
        for comp in COMPONENTS:
            if comp != "H2":
                product[comp] *= impurity_scale

    # Хвостовой газ = вход PSA - продукт PSA
    tail = {comp: F_psa_feed.get(comp, 0.0) - product.get(comp, 0.0) for comp in COMPONENTS}
    for comp in COMPONENTS:
        if tail[comp] < 0 and abs(tail[comp]) < 1e-10:
            tail[comp] = 0.0
        if tail[comp] < -1e-8:
            raise ValueError(f"Negative PSA tail flow for {comp}. Check parameters.")

    product_total = stream_total(product)
    product_purity = product["H2"] / product_total if product_total > 0 else 0.0
    tail_lhv_MW = lhv_power_MW(tail)

    info = {
        "H2_to_PSA_Nm3_h": h2_feed_nm3_h,
        "H2_product_Nm3_h": product["H2"] * VM_N,
        "H2_recovery_fraction": h2_recovery,
        "H2_recovery_percent": h2_recovery * 100.0,
        "H2_product_purity_mol_fraction": product_purity,
        "H2_product_purity_percent": product_purity * 100.0,
        "impurity_scale_to_meet_purity": impurity_scale,
        "tail_gas_Nm3_h": stream_total(tail) * VM_N,
        "tail_gas_LHV_MW": tail_lhv_MW,
        "P_product_bar": max(P_psa_feed_bar - PSA_PRODUCT_DP_BAR, 0.0),
        "P_tail_bar": PSA_TAIL_GAS_PRESSURE_BAR,
    }
    return product, tail, info

# =====================================================================
# 6. ОСНОВНОЙ ЗАПУСК
# =====================================================================


def main():
    if not INPUT_MATERIAL_BALANCE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_MATERIAL_BALANCE}. Run 03_final_balances.py first."
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_MATERIAL_BALANCE, encoding="utf-8-sig")
    F_lts, T_lts_C, P_lts_bar = load_stream_from_csv(df, "LTS_out")

    # Separator
    F_to_psa, F_condensate, sep_info = calculate_separator(F_lts, T_lts_C, P_lts_bar)

    # PSA
    psa_product, psa_tail, psa_info = calculate_psa(F_to_psa, sep_info["P_sep_bar"])

    # 04a: separator balance
    sep_rows = []
    sep_rows += stream_rows("separator_inlet_from_LTS", F_lts, T_lts_C, P_lts_bar)
    sep_rows += stream_rows("gas_to_PSA", F_to_psa, SEPARATOR_T_C, sep_info["P_sep_bar"])
    sep_rows += stream_rows("condensed_water", F_condensate, SEPARATOR_T_C, sep_info["P_sep_bar"])
    pd.DataFrame(sep_rows).to_csv(
        RESULTS_DIR / "04a_separator_balance.csv", index=False, encoding="utf-8-sig"
    )

    # 04b: PSA balance
    psa_rows = []
    psa_rows += stream_rows("PSA_feed", F_to_psa, SEPARATOR_T_C, sep_info["P_sep_bar"])
    psa_rows += stream_rows("H2_product", psa_product, SEPARATOR_T_C, psa_info["P_product_bar"])
    psa_rows += stream_rows("PSA_tail_gas", psa_tail, SEPARATOR_T_C, psa_info["P_tail_bar"])
    pd.DataFrame(psa_rows).to_csv(
        RESULTS_DIR / "04b_psa_balance.csv", index=False, encoding="utf-8-sig"
    )

    # 04c: summary
    summary = {
        "separator_T_C": sep_info["T_sep_C"],
        "separator_P_bar": sep_info["P_sep_bar"],
        "P_sat_H2O_at_60C_bar": sep_info["P_sat_H2O_bar"],
        "gas_to_PSA_H2O_mol_frac": F_to_psa["H2O"] / stream_total(F_to_psa),
        "H2O_condensed_kg_h": sep_info["H2O_condensed_kg_h"],
        "H2_to_PSA_Nm3_h": psa_info["H2_to_PSA_Nm3_h"],
        "target_H2_product_Nm3_h": TARGET_H2_PRODUCT_NM3_H,
        "actual_H2_product_Nm3_h": psa_info["H2_product_Nm3_h"],
        "required_H2_recovery_percent": psa_info["H2_recovery_percent"],
        "H2_product_purity_percent": psa_info["H2_product_purity_percent"],
        "H2_product_total_Nm3_h_with_impurities": stream_total(psa_product) * VM_N,
        "PSA_tail_gas_Nm3_h": psa_info["tail_gas_Nm3_h"],
        "PSA_tail_gas_LHV_MW": psa_info["tail_gas_LHV_MW"],
        "PSA_product_pressure_bar": psa_info["P_product_bar"],
        "PSA_tail_pressure_bar": psa_info["P_tail_bar"],
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "04c_final_product_and_tail_gas_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Step 04 complete: separator + PSA simple model.")
    print(f"Gas to PSA H2 = {psa_info['H2_to_PSA_Nm3_h']:.1f} Nm3/h")
    print(f"Target H2 product = {TARGET_H2_PRODUCT_NM3_H:.1f} Nm3/h")
    print(f"Required PSA H2 recovery = {psa_info['H2_recovery_percent']:.2f} %")
    print(f"H2 product purity = {psa_info['H2_product_purity_percent']:.4f} mol.%")
    print(f"Condensed water = {sep_info['H2O_condensed_kg_h']:.1f} kg/h")
    print(f"PSA tail gas = {psa_info['tail_gas_Nm3_h']:.1f} Nm3/h")
    print(f"PSA tail gas LHV = {psa_info['tail_gas_LHV_MW']:.3f} MW")
    print("CSV files are saved in:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
