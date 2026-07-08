# -*- coding: utf-8 -*-
"""
Core model for a 6500 Nm3/h H2 SMR-HTS-LTS reactor block.

The module is intentionally self-contained and editable. Main assumptions are
collected in the first sections. Units used in the code:
    flows       kmol/h
    pressure    bar
    temperature deg C in public functions, K inside kinetic equations
    heat duty   kJ/h and MW
    catalyst    kg

Process logic:
    SMR: tubular reactor, linear temperature profile, external heat input.
    HTS: adiabatic WGS reactor.
    LTS: adiabatic WGS reactor.
    Pressure drop: Ergun equation in the primary hydraulic calculation;
                   fixed pressure drops can be reused in operating optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

# =============================================================================
# 1. GLOBAL CONSTANTS
# =============================================================================

R = 8.314
EPS = 1.0e-12
NM3_PER_KMOL = 22.414
T_REF_C = 0.0

COMPONENTS = ["CH4", "H2O", "CO", "CO2", "H2", "N2"]
DRY_COMPONENTS = ["CH4", "CO", "CO2", "H2", "N2"]

MW = {
    "CH4": 16.043,
    "H2O": 18.015,
    "CO": 28.010,
    "CO2": 44.010,
    "H2": 2.016,
    "N2": 28.014,
}

# Mean heat capacities, kJ/(kmol*K). These are the same simplified values as in
# the uploaded diploma scripts, so the recalculation remains comparable.
CP = {
    "CH4": 55.0,
    "H2O": 40.0,
    "CO": 31.0,
    "CO2": 45.0,
    "H2": 29.0,
    "N2": 30.0,
}

# Reaction heat effects, kJ/kmol at standard state.
DH_SMR_1 = 206000.0    # CH4 + H2O  -> CO  + 3H2, endothermic
DH_SMR_2 = -41200.0    # CO  + H2O  -> CO2 + H2, exothermic
DH_SMR_3 = 165000.0    # CH4 + 2H2O -> CO2 + 4H2, endothermic
DH_WGS = -41200.0      # CO + H2O -> CO2 + H2

# =============================================================================
# 2. EDITABLE DESIGN BASIS
# =============================================================================

TARGET_H2_NM3_H = 6500.0
F_CH4_FEED_NM3_H = 2600.0

# If True, 2600 Nm3/h is interpreted as methane flow, and small natural gas
# impurities are added according to dry gas composition below. If False, feed is
# pure CH4 plus process steam.
INCLUDE_NG_IMPURITIES = True
Y_CH4_DRY = 0.960
Y_CO2_DRY = 0.005
Y_N2_DRY = 0.035
H2_FRACTION_OF_CH4_FEED = 0.001

T_FEED_GAS_C = 380.0
T_WATER_IN_C = 25.0
P_WATER_IN_BAR = 1.0
T_STEAM_GENERATION_C = 230.0
T_STEAM_TO_MIXER_C = 380.0
T_FINAL_COOLING_C = 60.0

# Water/steam heat-balance constants, approximate engineering values.
CP_WATER_LIQ_KJ_KG_K = 4.30
CP_STEAM_KJ_KG_K = 2.20
WATER_LATENT_230C_KJ_KG = 1830.0
WATER_LATENT_60C_KJ_KG = 2358.0
WATER_SPECIFIC_VOLUME_M3_KG = 0.00105
PUMP_EFFICIENCY = 0.75

# Baseline for the first catalyst/hydraulic calculation.
BASE_SC = 3.0
BASE_P_IN_BAR = 19.0
BASE_SMR_T_IN_C = 850.0
BASE_SMR_T_OUT_C = 750.0
BASE_HTS_T_IN_C = 350.0
BASE_LTS_T_IN_C = 220.0

# Pressure drops of coolers between stages. The user requested 0.5 bar.
DP_COOLER_BEFORE_HTS_BAR = 0.5
DP_COOLER_BEFORE_LTS_BAR = 0.5

MIN_PRODUCT_PRESSURE_BAR = 16.0
MIN_INITIAL_PRESSURE_BAR = 19.0
MAX_INITIAL_PRESSURE_BAR = 40.0
PRESSURE_STEP_BAR = 2.0

# =============================================================================
# 3. CATALYSTS, RESERVE FACTORS AND GEOMETRY
# =============================================================================

CAT_SMR = {
    "stage": "SMR",
    "name": "NIAP-03-01 type Ni/Al2O3-MgO catalyst",
    "d_pellet": 16.5e-3,
    "h_pellet": 14.0e-3,
    "d_hole": 3.1e-3,
    "n_holes": 7,
    "rho_bulk": 1000.0,
    "void_bed": 0.55,
    "eta_eff": 0.05,
    "K_res": 1.35,
}

CAT_HTS = {
    "stage": "HTS",
    "name": "NIAP-05-01 / STK-SMT Fe-Cr-Cu catalyst",
    "d_pellet": 9.0e-3,
    "h_pellet": 5.5e-3,
    "n_holes": 0,
    "rho_bulk": 1400.0,
    "void_bed": 0.40,
    "eta_eff": 0.70,
    "K_res": 1.90,
}

CAT_LTS = {
    "stage": "LTS",
    "name": "NTK-4 Cu-Zn-Cr-Al catalyst",
    "d_pellet": 5.0e-3,
    "h_pellet": 5.0e-3,
    "n_holes": 0,
    "rho_bulk": 1500.0,
    "void_bed": 0.50,
    "eta_eff": 0.85,
    "K_res": 2.50,
}

RESERVE_FACTOR_RANGES = {
    "SMR": (1.20, 1.50),
    "HTS": (1.75, 2.00),
    "LTS": (2.00, 3.00),
}

REACTOR_GEOM = {
    "smr_tubes_installed": 400,
    "smr_d_inner_m": 0.100,
    "smr_tube_len_m": 12.0,
    "hts_d_vessel_m": 1.6,
    "lts_d_vessel_m": 1.6,
}

AUTO_SELECT_ACTIVE_SMR_TUBES = True
TARGET_SMR_U_M_S = 2.0
MIN_SMR_U_M_S = 1.0
MAX_SMR_U_M_S = 5.0
MAX_SMR_DP_BAR = 3.0

# Vessel diameter selection for the adiabatic HTS/LTS reactors.
# If AUTO_SELECT_VESSEL_DIAMETER is True, the diameter of each shift vessel is
# sized so that the superficial gas velocity falls inside the window
# [MIN_WGS_U_M_S, MAX_WGS_U_M_S], targeting TARGET_WGS_U_M_S at the reactor
# inlet. The diameter is rounded to WGS_DIAMETER_STEP_M and clamped so the
# velocity stays inside the window. If False, the fixed diameters in
# REACTOR_GEOM ("hts_d_vessel_m" / "lts_d_vessel_m") are used instead.
AUTO_SELECT_VESSEL_DIAMETER = True
TARGET_WGS_U_M_S = 0.6
MIN_WGS_U_M_S = 0.3
MAX_WGS_U_M_S = 1.0
WGS_DIAMETER_STEP_M = 0.05

# Numerics. Increase layer counts for final checking if needed.
N_SMR_LAYERS = 100
N_HTS_LAYERS = 60
N_LTS_LAYERS = 70

# Temperature ranges for operating optimization.
T_SMR_IN_RANGE_C = (750.0, 920.0)
T_SMR_OUT_MIN_C = 740.0
T_SMR_DT_RANGE_C = (0.0, 300.0)
T_HTS_IN_RANGE_C = (320.0, 400.0)
T_HTS_OUT_RANGE_C = (370.0, 450.0)
T_LTS_IN_RANGE_C = (190.0, 260.0)
T_LTS_OUT_RANGE_C = (220.0, 280.0)
SC_RANGE = (2.0, 4.0)

for _cat in (CAT_SMR, CAT_HTS, CAT_LTS):
    # Filled below after function definition.
    pass

# =============================================================================
# 4. BASIC UTILITIES
# =============================================================================

def kmol_h_from_nm3_h(nm3_h: float) -> float:
    return nm3_h / NM3_PER_KMOL


def nm3_h_from_kmol_h(kmol_h: float) -> float:
    return kmol_h * NM3_PER_KMOL


def equivalent_diameter(cat: Dict[str, float]) -> float:
    """Sauter equivalent diameter dp = 6*V/A for cylindrical pellets."""
    d = float(cat["d_pellet"])
    h = float(cat["h_pellet"])
    V = 0.25 * math.pi * d**2 * h
    A = 2.0 * (0.25 * math.pi * d**2) + math.pi * d * h
    if int(cat.get("n_holes", 0)) > 0:
        dh = float(cat["d_hole"])
        n = int(cat["n_holes"])
        V -= n * 0.25 * math.pi * dh**2 * h
        A += n * math.pi * dh * h
    return 6.0 * V / max(A, EPS)


for _cat in (CAT_SMR, CAT_HTS, CAT_LTS):
    _cat["dp_eq"] = equivalent_diameter(_cat)


def total_flow(F: Dict[str, float]) -> float:
    return float(sum(F[c] for c in COMPONENTS) + EPS)


def dry_flow(F: Dict[str, float]) -> float:
    return float(sum(F[c] for c in DRY_COMPONENTS) + EPS)


def mole_fractions(F: Dict[str, float]) -> Dict[str, float]:
    ft = total_flow(F)
    return {c: F[c] / ft for c in COMPONENTS}


def partial_pressures_bar(F: Dict[str, float], P_bar: float) -> Dict[str, float]:
    y = mole_fractions(F)
    return {c: y[c] * P_bar for c in COMPONENTS}


def partial_pressures_kpa(F: Dict[str, float], P_bar: float) -> Dict[str, float]:
    p = partial_pressures_bar(F, P_bar)
    return {c: p[c] * 100.0 for c in COMPONENTS}


def clamp_stream(F: Dict[str, float]) -> Dict[str, float]:
    for c in COMPONENTS:
        if F[c] < EPS:
            F[c] = EPS
    return F


def limit_reaction_extent(F: Dict[str, float], nu: Dict[str, float], dxi: float) -> float:
    if dxi <= 0.0:
        return 0.0
    max_xi = dxi
    for c, coeff in nu.items():
        if coeff < 0.0:
            max_xi = min(max_xi, 0.999 * F[c] / abs(coeff))
    return max(max_xi, 0.0)


def heat_capacity_flow(F: Dict[str, float]) -> float:
    return float(sum(F[c] * CP[c] for c in COMPONENTS) + EPS)


def sensible_heat_change_kJ_h(F: Dict[str, float], T_from_C: float, T_to_C: float) -> float:
    return heat_capacity_flow(F) * (T_to_C - T_from_C)


def sensible_enthalpy_kJ_h(F: Dict[str, float], T_C: float, T_ref_C: float = T_REF_C) -> float:
    return heat_capacity_flow(F) * (T_C - T_ref_C)


def dry_co_percent(F: Dict[str, float]) -> float:
    return F["CO"] / dry_flow(F) * 100.0


def avg_mw_kg_kmol(F: Dict[str, float]) -> float:
    return sum(F[c] * MW[c] for c in COMPONENTS) / total_flow(F)


def mass_flow_kg_h(F: Dict[str, float]) -> float:
    return sum(F[c] * MW[c] for c in COMPONENTS)


def make_feed(SC: float = BASE_SC, include_impurities: bool = INCLUDE_NG_IMPURITIES) -> Dict[str, float]:
    """Create feed from 2600 Nm3/h CH4 and a chosen steam-to-carbon ratio."""
    f_ch4 = kmol_h_from_nm3_h(F_CH4_FEED_NM3_H)
    if include_impurities:
        f_dry = f_ch4 / Y_CH4_DRY
        f_co2 = Y_CO2_DRY * f_dry
        f_n2 = Y_N2_DRY * f_dry
    else:
        f_co2 = 0.0
        f_n2 = 0.0
    f_h2 = H2_FRACTION_OF_CH4_FEED * f_ch4
    return {
        "CH4": f_ch4,
        "H2O": SC * f_ch4,
        "CO": 0.0,
        "CO2": f_co2,
        "H2": f_h2,
        "N2": f_n2,
    }

# =============================================================================
# 5. PHYSICAL PROPERTIES AND ERGUN EQUATION
# =============================================================================

def gas_viscosity_pa_s(F: Dict[str, float], T_K: float) -> float:
    # Approximate mixture viscosity, microPa*s at 800 K with T^0.7 correction.
    mu_ref_micro = {
        "CH4": 25.0,
        "H2O": 28.0,
        "CO": 35.0,
        "CO2": 33.0,
        "H2": 18.0,
        "N2": 32.0,
    }
    y = mole_fractions(F)
    mu_micro = sum(y[c] * mu_ref_micro[c] * (T_K / 800.0) ** 0.7 for c in COMPONENTS)
    return mu_micro * 1.0e-6


def gas_density_kg_m3(F: Dict[str, float], T_K: float, P_bar: float) -> float:
    P_pa = max(P_bar, 0.01) * 1.0e5
    M_kg_mol = avg_mw_kg_kmol(F) / 1000.0
    return P_pa * M_kg_mol / (R * T_K)


def volumetric_flow_m3_s(F: Dict[str, float], T_K: float, P_bar: float) -> float:
    n_kmol_s = total_flow(F) / 3600.0
    return n_kmol_s * 1000.0 * R * T_K / (max(P_bar, 0.01) * 1.0e5)


def ergun_dp_dz_pa_m(
    F: Dict[str, float],
    T_K: float,
    P_bar: float,
    A_cross_m2: float,
    cat: Dict[str, float],
) -> Tuple[float, float]:
    """Return dP/dz in Pa/m and superficial velocity in m/s."""
    eps = float(cat["void_bed"])
    dp = float(cat["dp_eq"])
    rho = gas_density_kg_m3(F, T_K, P_bar)
    mu = gas_viscosity_pa_s(F, T_K)
    u_s = volumetric_flow_m3_s(F, T_K, P_bar) / max(A_cross_m2, EPS)
    term1 = 150.0 * mu * (1.0 - eps) ** 2 / (eps**3 * dp**2) * u_s
    term2 = 1.75 * rho * (1.0 - eps) / (eps**3 * dp) * u_s**2
    return term1 + term2, u_s


def select_active_smr_tubes(F: Dict[str, float], T_C: float, P_bar: float) -> int:
    installed = int(REACTOR_GEOM["smr_tubes_installed"])
    if not AUTO_SELECT_ACTIVE_SMR_TUBES:
        return installed
    A_tube = 0.25 * math.pi * REACTOR_GEOM["smr_d_inner_m"] ** 2
    vdot = volumetric_flow_m3_s(F, T_C + 273.15, P_bar)
    n = int(math.ceil(vdot / max(TARGET_SMR_U_M_S * A_tube, EPS)))
    return int(max(1, min(installed, n)))


def select_vessel_diameter(
    F: Dict[str, float],
    T_C: float,
    P_bar: float,
    target_u_m_s: float = TARGET_WGS_U_M_S,
    step_m: float = WGS_DIAMETER_STEP_M,
) -> float:
    """Size an adiabatic vessel diameter for a target superficial velocity.

    The diameter is computed from the inlet volumetric flow so that the inlet
    superficial velocity equals target_u_m_s, rounded to step_m, then clamped so
    the velocity stays inside [MIN_WGS_U_M_S, MAX_WGS_U_M_S]. Smaller diameter
    means higher velocity, so the clamp uses d_min at MAX_WGS_U_M_S and d_max at
    MIN_WGS_U_M_S.
    """
    vdot = volumetric_flow_m3_s(F, T_C + 273.15, P_bar)
    d_target = math.sqrt(4.0 * vdot / (math.pi * max(target_u_m_s, EPS)))
    if step_m and step_m > 0.0:
        d_target = round(d_target / step_m) * step_m
    d_min = math.sqrt(4.0 * vdot / (math.pi * MAX_WGS_U_M_S))
    d_max = math.sqrt(4.0 * vdot / (math.pi * MIN_WGS_U_M_S))
    return float(min(max(d_target, d_min), d_max))

# =============================================================================
# 6. KINETIC MODELS
# =============================================================================

def keq_xu_bar(T_K: float) -> Tuple[float, float, float]:
    K1 = math.exp(30.114 - 26830.0 / T_K)
    K2 = math.exp(-4.036 + 4400.0 / T_K)
    K3 = K1 * K2
    return K1, K2, K3


def keq_hh_kpa(T_K: float) -> Tuple[float, float, float]:
    K1 = 1.198e17 * math.exp(-26830.0 / T_K)
    K2 = 1.767e-2 * math.exp(4400.0 / T_K)
    K3 = 2.117e15 * math.exp(-22430.0 / T_K)
    return K1, K2, K3


def keq_wgs(T_K: float) -> float:
    return math.exp(-4.33 + 4577.8 / T_K)


def xu_froment_rates(F: Dict[str, float], T_K: float, P_bar: float) -> Tuple[float, float, float]:
    p = partial_pressures_bar(F, P_bar)
    pCH4 = max(p["CH4"], EPS)
    pH2O = max(p["H2O"], EPS)
    pCO = max(p["CO"], EPS)
    pCO2 = max(p["CO2"], EPS)
    pH2 = max(p["H2"], 1.0e-9)
    k1 = 4.225e15 * math.exp(-240100.0 / (R * T_K))
    k2 = 1.995e6 * math.exp(-67130.0 / (R * T_K))
    k3 = 1.020e15 * math.exp(-243900.0 / (R * T_K))
    K1, K2, K3 = keq_xu_bar(T_K)
    KCO = 8.23e-5 * math.exp(70650.0 / (R * T_K))
    KH2 = 6.12e-9 * math.exp(82900.0 / (R * T_K))
    KCH4 = 6.65e-4 * math.exp(38280.0 / (R * T_K))
    KH2O = 1.77e5 * math.exp(-88680.0 / (R * T_K))
    den = 1.0 + KCO * pCO + KH2 * pH2 + KCH4 * pCH4 + KH2O * pH2O / pH2
    r1 = (k1 / pH2**2.5) * (pCH4 * pH2O - pH2**3.0 * pCO / (K1 + EPS)) / den**2
    r2 = (k2 / pH2) * (pCO * pH2O - pH2 * pCO2 / (K2 + EPS)) / den**2
    r3 = (k3 / pH2**3.5) * (pCH4 * pH2O**2.0 - pH2**4.0 * pCO2 / (K3 + EPS)) / den**2
    return max(r1, 0.0), max(r2, 0.0), max(r3, 0.0)


def hou_hughes_rates(F: Dict[str, float], T_K: float, P_bar: float) -> Tuple[float, float, float]:
    p = partial_pressures_kpa(F, P_bar)
    pCH4 = max(p["CH4"], EPS)
    pH2O = max(p["H2O"], EPS)
    pCO = max(p["CO"], EPS)
    pCO2 = max(p["CO2"], EPS)
    pH2 = max(p["H2"], 1.0e-9)
    k1_s = 5.922e8 * math.exp(-209200.0 / (R * T_K))
    k2_s = 6.028e-4 * math.exp(-15400.0 / (R * T_K))  # E2 = 15.4 kJ/mol (Hou & Hughes 2001, Table 7)
    k3_s = 1.093e3 * math.exp(-109400.0 / (R * T_K))
    K1, K2, K3 = keq_hh_kpa(T_K)
    KCO = 5.127e-13 * math.exp(140000.0 / (R * T_K))
    KH2 = 5.68e-10 * math.exp(93400.0 / (R * T_K))
    KH2O = 9.251 * math.exp(-15900.0 / (R * T_K))
    den = 1.0 + KCO * pCO + KH2 * pH2**0.5 + KH2O * (pH2O / pH2)
    r1_s = k1_s * (pCH4 * pH2O**0.5 / pH2**1.25) * (1.0 - (pH2**3.0 * pCO) / (K1 * pCH4 * pH2O + EPS)) / den**2
    r2_s = k2_s * (pCO * pH2O**0.5 / pH2**0.5) * (1.0 - (pH2 * pCO2) / (K2 * pCO * pH2O + EPS)) / den**2
    r3_s = k3_s * (pCH4 * pH2O / pH2**1.75) * (1.0 - (pH2**4.0 * pCO2) / (K3 * pCH4 * pH2O**2.0 + EPS)) / den**2
    return max(r1_s * 3600.0, 0.0), max(r2_s * 3600.0, 0.0), max(r3_s * 3600.0, 0.0)


def podolski_kim_rate(F: Dict[str, float], T_K: float, P_bar: float) -> float:
    p = partial_pressures_bar(F, P_bar)
    pCO = max(p["CO"], EPS)
    pH2O = max(p["H2O"], EPS)
    pCO2 = max(p["CO2"], EPS)
    pH2 = max(p["H2"], EPS)
    k = 3.898e10 * math.exp(-122859.0 / (R * T_K))
    K = keq_wgs(T_K)
    KCO = 3.32e-2 * math.exp(12820.0 / (R * T_K))
    KH2O = 610.09 * math.exp(-26008.0 / (R * T_K))
    KCO2 = 9.156e-5 * math.exp(52476.0 / (R * T_K))
    den = 1.0 + KCO * pCO + KH2O * pH2O + KCO2 * pCO2
    r = k * KCO * KH2O * (pCO * pH2O - (pCO2 * pH2) / (K + EPS)) / den**2
    return max(r, 0.0)

# Amadeo-Laborde LTS constants.
LTS_Tm = 454.3
LTS_km = 3.22596
LTS_Ea = 47400.0
LTS_KCOm = 2.1811
LTS_dHCO = -910.0
LTS_KH2Om = 3.9477e-1
LTS_dHH2O = -1420.0
LTS_KCO2m = 4.6385e-3
LTS_dHCO2 = -24720.0
LTS_KH2m = 5.1320e-2
LTS_dHH2 = -14400.0


def amadeo_laborde_rate(F: Dict[str, float], T_K: float, P_bar: float) -> float:
    p = partial_pressures_bar(F, P_bar)
    pCO = max(p["CO"], EPS)
    pH2O = max(p["H2O"], EPS)
    pCO2 = max(p["CO2"], EPS)
    pH2 = max(p["H2"], EPS)
    def cf(E: float) -> float:
        return math.exp(-(E / R) * (1.0 / T_K - 1.0 / LTS_Tm))
    k = LTS_km * cf(LTS_Ea)
    KCO = LTS_KCOm * cf(LTS_dHCO)
    KH2O = LTS_KH2Om * cf(LTS_dHH2O)
    KCO2 = LTS_KCO2m * cf(LTS_dHCO2)
    KH2 = LTS_KH2m * cf(LTS_dHH2)
    K = keq_wgs(T_K)
    den = 1.0 + KCO * pCO + KH2O * pH2O + KCO2 * pCO2 + KH2 * pH2
    r = k * (pCO * pH2O - (pCO2 * pH2) / (K + EPS)) / den**2
    return max(r, 0.0)

SMR_RATE_FUNCTIONS: Dict[str, Callable[[Dict[str, float], float, float], Tuple[float, float, float]]] = {
    "Hou-Hughes": hou_hughes_rates,
    "Xu-Froment": xu_froment_rates,
}

# =============================================================================
# 7. REACTOR MODELS
# =============================================================================

@dataclass
class ReactorResult:
    F_out: Dict[str, float]
    T_out_K: float
    P_out_bar: float
    Q_reaction_kJ_h: float
    geom: Dict[str, float]
    extents: Dict[str, float]


def _pressure_mode_update(
    dp_mode: str,
    fixed_dp_bar: Optional[float],
    dP_layer_ergun_bar: float,
    n_layers: int,
) -> float:
    if dp_mode == "fixed":
        return float(fixed_dp_bar or 0.0) / n_layers
    return dP_layer_ergun_bar


def run_smr_reactor(
    F0_total: Dict[str, float],
    W_loaded_kg: float,
    model_name: str,
    T_in_C: float,
    T_out_C: float,
    P_in_bar: float,
    active_tubes: Optional[int] = None,
    use_eta: bool = True,
    dp_mode: str = "ergun",
    fixed_dp_bar: Optional[float] = None,
    n_layers: int = N_SMR_LAYERS,
) -> ReactorResult:
    if model_name not in SMR_RATE_FUNCTIONS:
        raise ValueError(f"Unknown SMR model: {model_name}")
    if T_out_C > T_in_C:
        raise ValueError("SMR outlet temperature cannot exceed inlet temperature in this model")
    rate_fn = SMR_RATE_FUNCTIONS[model_name]
    if active_tubes is None:
        active_tubes = select_active_smr_tubes(F0_total, 0.5 * (T_in_C + T_out_C), P_in_bar)
    n_tubes = max(1, int(active_tubes))
    A_tube = 0.25 * math.pi * REACTOR_GEOM["smr_d_inner_m"] ** 2
    F = {c: F0_total[c] / n_tubes for c in COMPONENTS}
    W_layer = W_loaded_kg / n_tubes / n_layers
    bed_length_m = W_loaded_kg / (CAT_SMR["rho_bulk"] * A_tube * n_tubes + EPS)
    layer_length_m = bed_length_m / n_layers
    eta = CAT_SMR["eta_eff"] if use_eta else 1.0
    T_in_K = T_in_C + 273.15
    T_out_K = T_out_C + 273.15
    P = float(P_in_bar)

    nu1 = {"CH4": -1.0, "H2O": -1.0, "CO": 1.0, "CO2": 0.0, "H2": 3.0, "N2": 0.0}
    nu2 = {"CH4": 0.0, "H2O": -1.0, "CO": -1.0, "CO2": 1.0, "H2": 1.0, "N2": 0.0}
    nu3 = {"CH4": -1.0, "H2O": -2.0, "CO": 0.0, "CO2": 1.0, "H2": 4.0, "N2": 0.0}

    Q_tube = 0.0
    ext1 = ext2 = ext3 = 0.0
    dP_total_bar = 0.0
    u_values: List[float] = []

    for i in range(n_layers):
        frac = (i + 0.5) / n_layers
        T_K = T_in_K + (T_out_K - T_in_K) * frac
        P = max(P, 0.05)
        r1, r2, r3 = rate_fn(F, T_K, P)
        dxi1 = limit_reaction_extent(F, nu1, r1 * W_layer * eta)
        for c in COMPONENTS:
            F[c] += nu1[c] * dxi1
        dxi2 = limit_reaction_extent(F, nu2, r2 * W_layer * eta)
        for c in COMPONENTS:
            F[c] += nu2[c] * dxi2
        dxi3 = limit_reaction_extent(F, nu3, r3 * W_layer * eta)
        for c in COMPONENTS:
            F[c] += nu3[c] * dxi3
        F = clamp_stream(F)
        Q_tube += dxi1 * DH_SMR_1 + dxi2 * DH_SMR_2 + dxi3 * DH_SMR_3
        ext1 += dxi1
        ext2 += dxi2
        ext3 += dxi3

        dP_ergun_bar = 0.0
        u_s = volumetric_flow_m3_s(F, T_K, P) / A_tube
        if dp_mode == "ergun" and bed_length_m > 0:
            dpdz_pa_m, u_s = ergun_dp_dz_pa_m(F, T_K, P, A_tube, CAT_SMR)
            dP_ergun_bar = dpdz_pa_m * layer_length_m / 1.0e5
        dP_layer_bar = _pressure_mode_update(dp_mode, fixed_dp_bar, dP_ergun_bar, n_layers)
        P -= dP_layer_bar
        dP_total_bar += dP_layer_bar
        u_values.append(u_s)

    F_out = {c: F[c] * n_tubes for c in COMPONENTS}
    Q_total = Q_tube * n_tubes
    u_avg = float(np.mean(u_values)) if u_values else 0.0
    geom = {
        "stage": "SMR",
        "active_tubes": n_tubes,
        "installed_tubes": REACTOR_GEOM["smr_tubes_installed"],
        "d_inner_m": REACTOR_GEOM["smr_d_inner_m"],
        "tube_len_m": REACTOR_GEOM["smr_tube_len_m"],
        "bed_length_m": bed_length_m,
        "bed_length_ok": bool(bed_length_m <= REACTOR_GEOM["smr_tube_len_m"]),
        "u_s_avg_m_s": u_avg,
        "u_s_min_m_s": float(np.min(u_values)) if u_values else u_avg,
        "u_s_max_m_s": float(np.max(u_values)) if u_values else u_avg,
        "velocity_ok": bool(MIN_SMR_U_M_S <= u_avg <= MAX_SMR_U_M_S),
        "dP_bar": dP_total_bar,
        "dP_ok": bool(dP_total_bar <= MAX_SMR_DP_BAR),
        "dp_mode": dp_mode,
    }
    extents = {"smr_r1_kmol_h": ext1 * n_tubes, "smr_r2_kmol_h": ext2 * n_tubes, "smr_r3_kmol_h": ext3 * n_tubes}
    return ReactorResult(F_out, T_out_K, P, Q_total, geom, extents)


def _run_wgs_reactor(
    F0_total: Dict[str, float],
    W_loaded_kg: float,
    T_in_C: float,
    P_in_bar: float,
    cat: Dict[str, float],
    d_vessel_m: float,
    rate_fn: Callable[[Dict[str, float], float, float], float],
    use_eta: bool,
    dp_mode: str,
    fixed_dp_bar: Optional[float],
    n_layers: int,
) -> ReactorResult:
    F = F0_total.copy()
    W_layer = W_loaded_kg / n_layers
    T_K = T_in_C + 273.15
    P = float(P_in_bar)
    eta = cat["eta_eff"] if use_eta else 1.0
    A_cross = 0.25 * math.pi * d_vessel_m**2
    bed_length_m = W_loaded_kg / (cat["rho_bulk"] * A_cross + EPS)
    layer_length_m = bed_length_m / n_layers
    nu = {"CH4": 0.0, "H2O": -1.0, "CO": -1.0, "CO2": 1.0, "H2": 1.0, "N2": 0.0}
    Q_total = 0.0
    extent = 0.0
    dP_total_bar = 0.0
    u_values: List[float] = []
    for _ in range(n_layers):
        P = max(P, 0.05)
        r = rate_fn(F, T_K, P)
        dxi = limit_reaction_extent(F, nu, r * W_layer * eta)
        for c in COMPONENTS:
            F[c] += nu[c] * dxi
        F = clamp_stream(F)
        Q_layer = dxi * DH_WGS
        Q_total += Q_layer
        extent += dxi
        # Adiabatic bed: exothermic Q_layer is negative, so T increases.
        T_K += -Q_layer / heat_capacity_flow(F)

        dP_ergun_bar = 0.0
        u_s = volumetric_flow_m3_s(F, T_K, P) / A_cross
        if dp_mode == "ergun" and bed_length_m > 0.0:
            dpdz_pa_m, u_s = ergun_dp_dz_pa_m(F, T_K, P, A_cross, cat)
            dP_ergun_bar = dpdz_pa_m * layer_length_m / 1.0e5
        dP_layer_bar = _pressure_mode_update(dp_mode, fixed_dp_bar, dP_ergun_bar, n_layers)
        P -= dP_layer_bar
        dP_total_bar += dP_layer_bar
        u_values.append(u_s)
    geom = {
        "stage": cat["stage"],
        "d_vessel_m": d_vessel_m,
        "bed_length_m": bed_length_m,
        "u_s_avg_m_s": float(np.mean(u_values)) if u_values else 0.0,
        "u_s_min_m_s": float(np.min(u_values)) if u_values else 0.0,
        "u_s_max_m_s": float(np.max(u_values)) if u_values else 0.0,
        "dP_bar": dP_total_bar,
        "dp_mode": dp_mode,
    }
    geom["velocity_ok"] = bool(MIN_WGS_U_M_S <= geom["u_s_avg_m_s"] <= MAX_WGS_U_M_S)
    return ReactorResult(F, T_K, P, Q_total, geom, {"wgs_kmol_h": extent})


def run_hts_reactor(
    F0_total: Dict[str, float],
    W_loaded_kg: float,
    T_in_C: float,
    P_in_bar: float,
    use_eta: bool = True,
    dp_mode: str = "ergun",
    fixed_dp_bar: Optional[float] = None,
    n_layers: int = N_HTS_LAYERS,
    d_vessel_m: Optional[float] = None,
) -> ReactorResult:
    if d_vessel_m is None:
        d_vessel_m = (
            select_vessel_diameter(F0_total, T_in_C, P_in_bar)
            if AUTO_SELECT_VESSEL_DIAMETER
            else REACTOR_GEOM["hts_d_vessel_m"]
        )
    return _run_wgs_reactor(
        F0_total, W_loaded_kg, T_in_C, P_in_bar, CAT_HTS,
        d_vessel_m, podolski_kim_rate,
        use_eta, dp_mode, fixed_dp_bar, n_layers,
    )


def run_lts_reactor(
    F0_total: Dict[str, float],
    W_loaded_kg: float,
    T_in_C: float,
    P_in_bar: float,
    use_eta: bool = True,
    dp_mode: str = "ergun",
    fixed_dp_bar: Optional[float] = None,
    n_layers: int = N_LTS_LAYERS,
    d_vessel_m: Optional[float] = None,
) -> ReactorResult:
    if d_vessel_m is None:
        d_vessel_m = (
            select_vessel_diameter(F0_total, T_in_C, P_in_bar)
            if AUTO_SELECT_VESSEL_DIAMETER
            else REACTOR_GEOM["lts_d_vessel_m"]
        )
    return _run_wgs_reactor(
        F0_total, W_loaded_kg, T_in_C, P_in_bar, CAT_LTS,
        d_vessel_m, amadeo_laborde_rate,
        use_eta, dp_mode, fixed_dp_bar, n_layers,
    )

# =============================================================================
# 8. FULL REACTOR BLOCK SIMULATION
# =============================================================================

def simulate_block(
    W_smr_loaded_kg: float,
    W_hts_loaded_kg: float,
    W_lts_loaded_kg: float,
    smr_model: str = "Hou-Hughes",
    T_smr_in_C: float = BASE_SMR_T_IN_C,
    T_smr_out_C: float = BASE_SMR_T_OUT_C,
    T_hts_in_C: float = BASE_HTS_T_IN_C,
    T_lts_in_C: float = BASE_LTS_T_IN_C,
    SC: float = BASE_SC,
    P_smr_in_bar: float = BASE_P_IN_BAR,
    use_eta: bool = True,
    dp_mode: str = "ergun",
    fixed_dps: Optional[Dict[str, float]] = None,
    active_tubes: Optional[int] = None,
) -> Dict[str, object]:
    fixed_dps = fixed_dps or {}
    F0 = make_feed(SC=SC)
    if active_tubes is None:
        active_tubes = select_active_smr_tubes(F0, 0.5 * (T_smr_in_C + T_smr_out_C), P_smr_in_bar)
    smr = run_smr_reactor(
        F0, W_smr_loaded_kg, smr_model, T_smr_in_C, T_smr_out_C, P_smr_in_bar,
        active_tubes=active_tubes, use_eta=use_eta, dp_mode=dp_mode,
        fixed_dp_bar=fixed_dps.get("SMR"),
    )
    P_hts_in = smr.P_out_bar - DP_COOLER_BEFORE_HTS_BAR
    hts = run_hts_reactor(
        smr.F_out, W_hts_loaded_kg, T_hts_in_C, P_hts_in,
        use_eta=use_eta, dp_mode=dp_mode, fixed_dp_bar=fixed_dps.get("HTS"),
    )
    P_lts_in = hts.P_out_bar - DP_COOLER_BEFORE_LTS_BAR
    lts = run_lts_reactor(
        hts.F_out, W_lts_loaded_kg, T_lts_in_C, P_lts_in,
        use_eta=use_eta, dp_mode=dp_mode, fixed_dp_bar=fixed_dps.get("LTS"),
    )

    F_lts = lts.F_out
    H2_net_kmol_h = F_lts["H2"] - F0["H2"]
    H2_net_Nm3_h = nm3_h_from_kmol_h(H2_net_kmol_h)
    H2_specific = H2_net_Nm3_h / max(F_CH4_FEED_NM3_H, EPS)
    X_CH4_pct = (F0["CH4"] - F_lts["CH4"]) / max(F0["CH4"], EPS) * 100.0
    X_CO_HTS_pct = (smr.F_out["CO"] - hts.F_out["CO"]) / max(smr.F_out["CO"], EPS) * 100.0
    X_CO_LTS_pct = (hts.F_out["CO"] - lts.F_out["CO"]) / max(hts.F_out["CO"], EPS) * 100.0

    dP_total = P_smr_in_bar - lts.P_out_bar
    P_required_for_16 = MIN_PRODUCT_PRESSURE_BAR + dP_total
    result = {
        "smr_model": smr_model,
        "F0": F0,
        "F_smr": smr.F_out,
        "F_hts": hts.F_out,
        "F_lts": F_lts,
        "T_smr_in_C": T_smr_in_C,
        "T_smr_out_C": T_smr_out_C,
        "smr_delta_T_C": T_smr_in_C - T_smr_out_C,
        "T_hts_in_C": T_hts_in_C,
        "T_hts_out_C": hts.T_out_K - 273.15,
        "T_lts_in_C": T_lts_in_C,
        "T_lts_out_C": lts.T_out_K - 273.15,
        "SC": SC,
        "P_smr_in_bar": P_smr_in_bar,
        "P_smr_out_bar": smr.P_out_bar,
        "P_hts_in_bar": P_hts_in,
        "P_hts_out_bar": hts.P_out_bar,
        "P_lts_in_bar": P_lts_in,
        "P_lts_out_bar": lts.P_out_bar,
        "dP_total_bar": dP_total,
        "P_required_for_16bar_product": P_required_for_16,
        "W_smr_loaded_kg": W_smr_loaded_kg,
        "W_hts_loaded_kg": W_hts_loaded_kg,
        "W_lts_loaded_kg": W_lts_loaded_kg,
        "active_smr_tubes": active_tubes,
        "X_CH4_pct": X_CH4_pct,
        "X_CO_HTS_pct": X_CO_HTS_pct,
        "X_CO_LTS_pct": X_CO_LTS_pct,
        "CO_dry_pct": dry_co_percent(F_lts),
        "H2_net_kmol_h": H2_net_kmol_h,
        "H2_net_Nm3_h": H2_net_Nm3_h,
        "H2_specific_Nm3_per_Nm3_CH4": H2_specific,
        "Q_SMR_reaction_kJ_h": smr.Q_reaction_kJ_h,
        "Q_HTS_reaction_kJ_h": hts.Q_reaction_kJ_h,
        "Q_LTS_reaction_kJ_h": lts.Q_reaction_kJ_h,
        "Q_SMR_reaction_MW": smr.Q_reaction_kJ_h / 3.6e6,
        "Q_HTS_reaction_MW": hts.Q_reaction_kJ_h / 3.6e6,
        "Q_LTS_reaction_MW": lts.Q_reaction_kJ_h / 3.6e6,
        "geom_smr": smr.geom,
        "geom_hts": hts.geom,
        "geom_lts": lts.geom,
        "ext_smr": smr.extents,
        "ext_hts": hts.extents,
        "ext_lts": lts.extents,
    }
    result["feasible"] = bool(
        result["X_CH4_pct"] >= 90.0
        and result["CO_dry_pct"] <= 1.0
        and result["H2_specific_Nm3_per_Nm3_CH4"] >= 2.5
        and result["P_lts_out_bar"] >= MIN_PRODUCT_PRESSURE_BAR
        and result["geom_smr"]["dP_ok"]
        and result["geom_smr"]["velocity_ok"]
        and result["geom_smr"]["bed_length_ok"]
    )
    return result


def flatten_result(res: Dict[str, object]) -> Dict[str, float]:
    gs = res["geom_smr"]
    gh = res["geom_hts"]
    gl = res["geom_lts"]
    return {
        "smr_model": res["smr_model"],
        "T_smr_in_C": res["T_smr_in_C"],
        "T_smr_out_C": res["T_smr_out_C"],
        "smr_delta_T_C": res["smr_delta_T_C"],
        "T_hts_in_C": res["T_hts_in_C"],
        "T_hts_out_C": res["T_hts_out_C"],
        "T_lts_in_C": res["T_lts_in_C"],
        "T_lts_out_C": res["T_lts_out_C"],
        "SC": res["SC"],
        "P_smr_in_bar": res["P_smr_in_bar"],
        "P_lts_out_bar": res["P_lts_out_bar"],
        "dP_total_bar": res["dP_total_bar"],
        "P_required_for_16bar_product": res["P_required_for_16bar_product"],
        "W_smr_loaded_kg": res["W_smr_loaded_kg"],
        "W_hts_loaded_kg": res["W_hts_loaded_kg"],
        "W_lts_loaded_kg": res["W_lts_loaded_kg"],
        "active_smr_tubes": res["active_smr_tubes"],
        "X_CH4_pct": res["X_CH4_pct"],
        "X_CO_HTS_pct": res["X_CO_HTS_pct"],
        "X_CO_LTS_pct": res["X_CO_LTS_pct"],
        "CO_dry_pct": res["CO_dry_pct"],
        "H2_net_Nm3_h": res["H2_net_Nm3_h"],
        "H2_specific_Nm3_per_Nm3_CH4": res["H2_specific_Nm3_per_Nm3_CH4"],
        "Q_SMR_reaction_MW": res["Q_SMR_reaction_MW"],
        "Q_HTS_reaction_MW": res["Q_HTS_reaction_MW"],
        "Q_LTS_reaction_MW": res["Q_LTS_reaction_MW"],
        "SMR_dP_bar": gs["dP_bar"],
        "HTS_dP_bar": gh["dP_bar"],
        "LTS_dP_bar": gl["dP_bar"],
        "SMR_u_avg_m_s": gs["u_s_avg_m_s"],
        "HTS_u_avg_m_s": gh["u_s_avg_m_s"],
        "LTS_u_avg_m_s": gl["u_s_avg_m_s"],
        "SMR_bed_length_m": gs["bed_length_m"],
        "HTS_bed_length_m": gh["bed_length_m"],
        "LTS_bed_length_m": gl["bed_length_m"],
        "HTS_d_vessel_m": gh["d_vessel_m"],
        "LTS_d_vessel_m": gl["d_vessel_m"],
        "HTS_velocity_ok": gh.get("velocity_ok", None),
        "LTS_velocity_ok": gl.get("velocity_ok", None),
        "SMR_velocity_ok": gs["velocity_ok"],
        "SMR_dP_ok": gs["dP_ok"],
        "SMR_bed_length_ok": gs["bed_length_ok"],
        "feasible": res["feasible"],
    }

# =============================================================================
# 9. PRESSURE, HEAT AND MATERIAL BALANCE HELPERS
# =============================================================================

def ceil_to_step(x: float, step: float) -> float:
    return math.ceil(x / step - 1.0e-12) * step


def find_min_pressure_for_case(
    W_smr_loaded_kg: float,
    W_hts_loaded_kg: float,
    W_lts_loaded_kg: float,
    smr_model: str,
    T_smr_in_C: float,
    T_smr_out_C: float,
    T_hts_in_C: float,
    T_lts_in_C: float,
    SC: float,
    active_tubes: Optional[int],
    use_eta: bool = True,
    dp_mode: str = "ergun",
    fixed_dps: Optional[Dict[str, float]] = None,
    P_start_bar: float = MIN_INITIAL_PRESSURE_BAR,
    P_stop_bar: float = MAX_INITIAL_PRESSURE_BAR,
    P_step_bar: float = PRESSURE_STEP_BAR,
) -> Tuple[float, Dict[str, object]]:
    P = P_start_bar
    last = None
    while P <= P_stop_bar + 1.0e-9:
        res = simulate_block(
            W_smr_loaded_kg, W_hts_loaded_kg, W_lts_loaded_kg, smr_model,
            T_smr_in_C, T_smr_out_C, T_hts_in_C, T_lts_in_C, SC, P,
            use_eta=use_eta, dp_mode=dp_mode, fixed_dps=fixed_dps, active_tubes=active_tubes,
        )
        last = res
        if res["P_lts_out_bar"] >= MIN_PRODUCT_PRESSURE_BAR:
            return P, res
        # Jump close to the requirement and then continue with grid step.
        need = MIN_PRODUCT_PRESSURE_BAR - res["P_lts_out_bar"]
        if need > P_step_bar:
            P = ceil_to_step(P + need, P_step_bar)
        else:
            P += P_step_bar
    return P_stop_bar, last if last is not None else {}


def water_psat_bar_antoine(T_C: float) -> float:
    # Antoine equation for water, mmHg, valid near 1-100 C.
    A, B, C = 8.07131, 1730.63, 233.426
    return (10.0 ** (A - B / (C + T_C))) * 0.00133322


def condensation_at_temperature(F: Dict[str, float], T_C: float, P_bar: float) -> Dict[str, float]:
    F_dry = dry_flow(F)
    psat = min(water_psat_bar_antoine(T_C), 0.99 * P_bar)
    y_h2o_sat = psat / max(P_bar, EPS)
    h2o_vapor_max = y_h2o_sat / max(1.0 - y_h2o_sat, EPS) * F_dry
    h2o_vapor = min(F["H2O"], h2o_vapor_max)
    h2o_cond = max(F["H2O"] - h2o_vapor, 0.0)
    F_cool = F.copy()
    F_cool["H2O"] = h2o_vapor
    return {
        "H2O_vapor_kmol_h": h2o_vapor,
        "H2O_condensed_kmol_h": h2o_cond,
        "H2O_condensed_kg_h": h2o_cond * MW["H2O"],
        "F_cool": F_cool,
        "water_psat_bar": psat,
    }


def heat_balance_tables(res: Dict[str, object], final_cooling_T_C: float = T_FINAL_COOLING_C) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    F0 = res["F0"]
    F_gas_only = F0.copy()
    F_gas_only["H2O"] = 0.0
    F_steam_only = {c: 0.0 for c in COMPONENTS}
    F_steam_only["H2O"] = F0["H2O"]
    m_water_kg_h = F0["H2O"] * MW["H2O"]
    P_in = float(res["P_smr_in_bar"])

    Q_water_compression = (
        m_water_kg_h * WATER_SPECIFIC_VOLUME_M3_KG * max(P_in - P_WATER_IN_BAR, 0.0) * 1.0e5
        / max(PUMP_EFFICIENCY, EPS) / 1000.0
    )
    Q_water_heat = m_water_kg_h * CP_WATER_LIQ_KJ_KG_K * (T_STEAM_GENERATION_C - T_WATER_IN_C)
    Q_water_evap = m_water_kg_h * WATER_LATENT_230C_KJ_KG
    Q_steam_superheat = m_water_kg_h * CP_STEAM_KJ_KG_K * (T_STEAM_TO_MIXER_C - T_STEAM_GENERATION_C)
    Q_gas_380_to_smr = sensible_heat_change_kJ_h(F_gas_only, T_FEED_GAS_C, res["T_smr_in_C"])
    Q_steam_380_to_smr = sensible_heat_change_kJ_h(F_steam_only, T_STEAM_TO_MIXER_C, res["T_smr_in_C"])
    Q_mixture_380_to_smr = Q_gas_380_to_smr + Q_steam_380_to_smr

    Q_smr_reaction = float(res["Q_SMR_reaction_kJ_h"])
    Q_smr_sensible = sensible_enthalpy_kJ_h(res["F_smr"], res["T_smr_out_C"]) - sensible_enthalpy_kJ_h(F0, res["T_smr_in_C"])
    Q_smr_total = Q_smr_reaction + Q_smr_sensible

    Q_cool_smr_to_hts = -min(sensible_heat_change_kJ_h(res["F_smr"], res["T_smr_out_C"], res["T_hts_in_C"]), 0.0)
    Q_cool_hts_to_lts = -min(sensible_heat_change_kJ_h(res["F_hts"], res["T_hts_out_C"], res["T_lts_in_C"]), 0.0)
    Q_cool_lts_to_60_sens = -min(sensible_heat_change_kJ_h(res["F_lts"], res["T_lts_out_C"], final_cooling_T_C), 0.0)
    cond = condensation_at_temperature(res["F_lts"], final_cooling_T_C, res["P_lts_out_bar"])
    Q_cond = cond["H2O_condensed_kg_h"] * WATER_LATENT_60C_KJ_KG

    heat_rows = [
        ("water_compression_1bar_to_process_pressure", Q_water_compression, "liquid water pressurization from 1 bar"),
        ("water_heating_25_to_230C", Q_water_heat, "liquid water heating"),
        ("water_evaporation_at_230C", Q_water_evap, "steam generation"),
        ("steam_superheating_230_to_380C", Q_steam_superheat, "steam superheating"),
        ("raw_gas_heating_380_to_SMR_in", Q_gas_380_to_smr, "methane-rich gas part only"),
        ("steam_heating_380_to_SMR_in", Q_steam_380_to_smr, "steam part only"),
        ("steam_gas_mixture_heating_380_to_SMR_in", Q_mixture_380_to_smr, "sum of the previous two rows"),
        ("SMR_reaction_heat", Q_smr_reaction, "positive means heat demand"),
        ("SMR_sensible_change_inside_reactor", Q_smr_sensible, "outlet enthalpy minus inlet enthalpy"),
        ("SMR_total_reactor_heat", Q_smr_total, "reaction plus sensible change"),
        ("cooling_after_SMR_to_HTS_in", Q_cool_smr_to_hts, "interstage cooler before HTS"),
        ("HTS_adiabatic_reaction_heat", res["Q_HTS_reaction_kJ_h"], "negative means heat release inside bed"),
        ("cooling_after_HTS_to_LTS_in", Q_cool_hts_to_lts, "interstage cooler before LTS"),
        ("LTS_adiabatic_reaction_heat", res["Q_LTS_reaction_kJ_h"], "negative means heat release inside bed"),
        ("cooling_after_LTS_to_60C_sensible", Q_cool_lts_to_60_sens, "final sensible cooling"),
        ("cooling_after_LTS_to_60C_condensation", Q_cond, "water condensation at final cooler"),
    ]
    df_heat = pd.DataFrame([
        {"duty": name, "Q_kJ_h": q, "Q_MW": q / 3.6e6, "note": note}
        for name, q, note in heat_rows
    ])

    material_rows = []
    streams = [
        ("feed_to_SMR", F0, res["T_smr_in_C"], res["P_smr_in_bar"]),
        ("SMR_out", res["F_smr"], res["T_smr_out_C"], res["P_smr_out_bar"]),
        ("HTS_out", res["F_hts"], res["T_hts_out_C"], res["P_hts_out_bar"]),
        ("LTS_out", res["F_lts"], res["T_lts_out_C"], res["P_lts_out_bar"]),
        ("cooled_to_60C_gas", cond["F_cool"], final_cooling_T_C, res["P_lts_out_bar"]),
    ]
    for stream, F, T_C, P_bar in streams:
        ft = total_flow(F)
        fd = dry_flow(F)
        for c in COMPONENTS:
            material_rows.append({
                "stream": stream,
                "T_C": T_C,
                "P_bar": P_bar,
                "component": c,
                "kmol_h": F[c],
                "kg_h": F[c] * MW[c],
                "Nm3_h": nm3_h_from_kmol_h(F[c]),
                "wet_mol_frac": F[c] / ft,
                "dry_mol_frac": np.nan if c == "H2O" else F[c] / fd,
            })
        material_rows.append({
            "stream": stream,
            "T_C": T_C,
            "P_bar": P_bar,
            "component": "TOTAL",
            "kmol_h": ft,
            "kg_h": mass_flow_kg_h(F),
            "Nm3_h": nm3_h_from_kmol_h(ft),
            "wet_mol_frac": 1.0,
            "dry_mol_frac": 1.0,
        })
    material_rows.append({
        "stream": "final_cooling",
        "T_C": final_cooling_T_C,
        "P_bar": res["P_lts_out_bar"],
        "component": "H2O_condensed",
        "kmol_h": cond["H2O_condensed_kmol_h"],
        "kg_h": cond["H2O_condensed_kg_h"],
        "Nm3_h": nm3_h_from_kmol_h(cond["H2O_condensed_kmol_h"]),
        "wet_mol_frac": np.nan,
        "dry_mol_frac": np.nan,
    })
    df_mat = pd.DataFrame(material_rows)

    gs, gh, gl = res["geom_smr"], res["geom_hts"], res["geom_lts"]
    df_pressure = pd.DataFrame([
        {"item": "P_SMR_in", "value": res["P_smr_in_bar"], "unit": "bar"},
        {"item": "dP_SMR", "value": gs["dP_bar"], "unit": "bar"},
        {"item": "P_SMR_out", "value": res["P_smr_out_bar"], "unit": "bar"},
        {"item": "dP_cooler_before_HTS", "value": DP_COOLER_BEFORE_HTS_BAR, "unit": "bar"},
        {"item": "P_HTS_in", "value": res["P_hts_in_bar"], "unit": "bar"},
        {"item": "dP_HTS", "value": gh["dP_bar"], "unit": "bar"},
        {"item": "P_HTS_out", "value": res["P_hts_out_bar"], "unit": "bar"},
        {"item": "dP_cooler_before_LTS", "value": DP_COOLER_BEFORE_LTS_BAR, "unit": "bar"},
        {"item": "P_LTS_in", "value": res["P_lts_in_bar"], "unit": "bar"},
        {"item": "dP_LTS", "value": gl["dP_bar"], "unit": "bar"},
        {"item": "P_LTS_out", "value": res["P_lts_out_bar"], "unit": "bar"},
        {"item": "minimum_required_P_SMR_in_for_16bar_product", "value": res["P_required_for_16bar_product"], "unit": "bar"},
        {"item": "SMR_active_tubes", "value": gs["active_tubes"], "unit": "pcs"},
        {"item": "SMR_velocity_avg", "value": gs["u_s_avg_m_s"], "unit": "m/s"},
        {"item": "SMR_bed_length", "value": gs["bed_length_m"], "unit": "m"},
        {"item": "HTS_d_vessel", "value": gh["d_vessel_m"], "unit": "m"},
        {"item": "HTS_velocity_avg", "value": gh["u_s_avg_m_s"], "unit": "m/s"},
        {"item": "HTS_bed_length", "value": gh["bed_length_m"], "unit": "m"},
        {"item": "LTS_d_vessel", "value": gl["d_vessel_m"], "unit": "m"},
        {"item": "LTS_velocity_avg", "value": gl["u_s_avg_m_s"], "unit": "m/s"},
        {"item": "LTS_bed_length", "value": gl["bed_length_m"], "unit": "m"},
    ])

    df_summary = pd.DataFrame([{
        "target_H2_Nm3_h": TARGET_H2_NM3_H,
        "feed_CH4_Nm3_h": F_CH4_FEED_NM3_H,
        "SC": res["SC"],
        "H2_net_Nm3_h": res["H2_net_Nm3_h"],
        "H2_specific_Nm3_per_Nm3_CH4": res["H2_specific_Nm3_per_Nm3_CH4"],
        "X_CH4_pct": res["X_CH4_pct"],
        "CO_dry_pct": res["CO_dry_pct"],
        "P_lts_out_bar": res["P_lts_out_bar"],
        "T_smr_in_C": res["T_smr_in_C"],
        "T_smr_out_C": res["T_smr_out_C"],
        "T_hts_in_C": res["T_hts_in_C"],
        "T_hts_out_C": res["T_hts_out_C"],
        "T_lts_in_C": res["T_lts_in_C"],
        "T_lts_out_C": res["T_lts_out_C"],
        "SMR_total_reactor_heat_MW": Q_smr_total / 3.6e6,
        "steam_generation_total_MW": (Q_water_compression + Q_water_heat + Q_water_evap + Q_steam_superheat) / 3.6e6,
        "mixture_heating_380_to_SMR_in_MW": Q_mixture_380_to_smr / 3.6e6,
        "cooling_total_MW": (Q_cool_smr_to_hts + Q_cool_hts_to_lts + Q_cool_lts_to_60_sens + Q_cond) / 3.6e6,
        "water_to_steam_kg_h": m_water_kg_h,
        "H2O_condensed_kg_h_after_LTS": cond["H2O_condensed_kg_h"],
    }])
    return df_mat, df_heat, df_pressure, df_summary


def atom_balance(F_in: Dict[str, float], F_out: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    atoms_in = {
        "C": F_in["CH4"] + F_in["CO"] + F_in["CO2"],
        "H": 4 * F_in["CH4"] + 2 * F_in["H2O"] + 2 * F_in["H2"],
        "O": F_in["H2O"] + F_in["CO"] + 2 * F_in["CO2"],
        "N": 2 * F_in["N2"],
    }
    atoms_out = {
        "C": F_out["CH4"] + F_out["CO"] + F_out["CO2"],
        "H": 4 * F_out["CH4"] + 2 * F_out["H2O"] + 2 * F_out["H2"],
        "O": F_out["H2O"] + F_out["CO"] + 2 * F_out["CO2"],
        "N": 2 * F_out["N2"],
    }
    return {k: (atoms_in[k], atoms_out[k]) for k in atoms_in}


def atom_balance_table(F_in: Dict[str, float], F_out: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for atom, (vin, vout) in atom_balance(F_in, F_out).items():
        rows.append({
            "atom": atom,
            "in_kmol_atoms_h": vin,
            "out_kmol_atoms_h": vout,
            "difference": vout - vin,
            "relative_error_pct": (vout - vin) / max(abs(vin), EPS) * 100.0,
        })
    return pd.DataFrame(rows)


def results_dir(path: str | Path = "results") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
