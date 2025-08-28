# -*- coding: utf-8 -*-
"""
Economic analysis for lithium-mediated electrochemical ammonia synthesis.

This script reuses the utility functions from the direct NRR model but
includes the cost of lithium consumption.  When executed it generates two
plots saved in the working directory:

- ``lcoa_vs_j.png``: LCOA as a function of total current density.
- ``parity_vs_price.png``: current density required to reach a target
  LCOA as a function of electricity price for different Faradaic
  efficiencies.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for figure generation
import matplotlib.pyplot as plt

# ---------- Physical constants ----------
F = 96485.0      # C/mol
M_NH3 = 0.01703  # kg/mol
z = 6.0

# ---------- Economic utilities ----------
def crf(i: float, N: int) -> float:
    """Capital Recovery Factor."""
    if i <= 0:
        return 1.0 / max(N, 1)
    return (i * (1 + i) ** N) / ((1 + i) ** N - 1)


def annual_worth_replacement(cost: float, i: float, k_years: int) -> float:
    """Equivalent annual cost of periodic replacement."""
    if k_years <= 0 or cost <= 0:
        return 0.0
    if i <= 0:
        return cost / k_years
    return cost * (i) / (1.0 - (1.0 + i) ** (-k_years))


# ---------- Degradation models ----------
def avg_factor_exponential(d: float, N: int) -> float:
    """Average factor for exponential decay ``d`` over ``N`` years."""
    if d <= 0:
        return 1.0
    return (1.0 - (1.0 - d) ** N) / (N * d)


# ---------- Core calculations ----------
def production_annual(eta_fe, j_tot, A, t_op_h_per_y):
    """Annual production (kg/y) from Faraday's law."""
    mol_per_s = (eta_fe * j_tot * A) / (z * F)
    kg_per_s = mol_per_s * M_NH3
    return kg_per_s * 3600.0 * t_op_h_per_y


def e_specific_kwh_per_kg(U_cell, eta_fe):
    """Specific energy consumption [kWh/kg]."""
    K = (z * F / M_NH3) / 3.6e6
    return K * (U_cell / np.maximum(eta_fe, 1e-12))


def power_kw(j_tot, A, U_cell):
    """Electric power [kW]."""
    return (j_tot * A * U_cell) / 1000.0


def lcoa_levelized_lithium(params):
    """LCOA including lithium consumption costs.

    Parameters mirror those used for the direct NRR model with two
    additional keys:
    - ``Li_loss_kg_per_kgNH3``: lithium consumed per kg NH3 produced.
    - ``c_Li_per_kg``: cost of lithium in €/kg.
    """
    eta0 = params["eta_fe"]
    j0 = params["j_tot"]
    U_cell = params["U_cell"]
    A = params["A"]
    c_A = params["c_A"]
    c_P = params["c_P"]
    c_el_MWh = params["c_el_MWh"]
    f_OandM = params["f_OandM"]
    i = params["i"]
    N = params["N"]
    t_op = params["t_op"]

    # Balance of plant
    cA_bop = params.get("cA_BoP_extra", 0.0)
    cP_bop = params.get("cP_BoP_extra", 0.0)
    opex_bop = params.get("OPEX_BoP_extra_year", 0.0)

    # Availability and degradation
    availability = params.get("availability", 1.0)
    t_op_eff = t_op * availability
    d_eta = params.get("deg_eta_yearly", 0.0)
    d_j = params.get("deg_j_yearly", 0.0)
    f_eta_avg = avg_factor_exponential(d_eta, N)
    f_j_avg = avg_factor_exponential(d_j, N)
    eta_eff = eta0 * f_eta_avg
    j_eff = j0 * f_j_avg

    # CAPEX including BoP
    P_kw_eff = power_kw(j_eff, A, U_cell)
    CAPEX_area = (c_A + cA_bop) * A
    CAPEX_power = (c_P + cP_bop) * P_kw_eff
    CAPEX_total = CAPEX_area + CAPEX_power

    # Periodic replacements
    rep_every = params.get("rep_interval_years", 0)
    rep_fracA = params.get("rep_frac_of_CAPEX_area", 0.0)
    rep_fracP = params.get("rep_frac_of_CAPEX_power", 0.0)
    rep_cost = rep_fracA * CAPEX_area + rep_fracP * CAPEX_power
    AW_rep = annual_worth_replacement(rep_cost, i, rep_every)

    # OPEX
    CRF = crf(i, N)
    OPEX_OandM = f_OandM * CAPEX_total
    m_anual = production_annual(eta_eff, j_eff, A, t_op_eff)
    E_spec = e_specific_kwh_per_kg(U_cell, eta_eff)
    OPEX_el = E_spec * m_anual * (c_el_MWh / 1000.0)

    # Lithium consumption cost
    Li_loss = params.get("Li_loss_kg_per_kgNH3", 0.0)
    c_Li = params.get("c_Li_per_kg", 0.0)
    OPEX_Li = Li_loss * c_Li * m_anual

    annual_cost = CRF * CAPEX_total + OPEX_OandM + OPEX_el + opex_bop + AW_rep + OPEX_Li
    lcoa = annual_cost / max(m_anual, 1e-12)

    breakdown = {
        "CRF": CRF,
        "CAPEX_area": CAPEX_area,
        "CAPEX_power": CAPEX_power,
        "CAPEX_total": CAPEX_total,
        "OPEX_OandM": OPEX_OandM,
        "OPEX_el": OPEX_el,
        "OPEX_BoP_extra_year": opex_bop,
        "AW_replacement": AW_rep,
        "OPEX_Li": OPEX_Li,
        "E_spec_kWh_per_kg": E_spec,
        "m_anual_kg": m_anual,
        "annual_cost_eur": annual_cost,
        "P_kw_eff": P_kw_eff,
        "eta_eff": eta_eff,
        "j_eff": j_eff,
        "t_op_eff": t_op_eff,
    }
    return lcoa, breakdown


def parity_j_for_eta_lithium(eta, c_el_MWh, params, cB, j_lo=50.0, j_hi=50000.0, tol=1e-6, maxit=100):
    """Find ``j_tot`` such that LCOA equals ``cB`` for given ``eta`` and electricity price."""
    pp = dict(params)
    pp["eta_fe"] = float(eta)
    pp["c_el_MWh"] = float(c_el_MWh)

    pp["j_tot"] = j_lo
    f_lo, _ = lcoa_levelized_lithium(pp)
    pp["j_tot"] = j_hi
    f_hi, _ = lcoa_levelized_lithium(pp)

    f_lo -= cB
    f_hi -= cB

    if f_lo <= 0.0:
        return j_lo
    if f_hi > 0.0:
        return np.nan

    lo, hi = j_lo, j_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        pp["j_tot"] = mid
        f_mid, _ = lcoa_levelized_lithium(pp)
        f_mid -= cB
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) / (1.0 + mid) < tol:
            return hi
    return hi


# ---------- Plot helpers ----------
def plot_lcoa_vs_j(params, j_values, output):
    """Plot LCOA as a function of ``j_tot`` and save to ``output``."""
    lcoas = []
    for j in j_values:
        pp = dict(params)
        pp["j_tot"] = j
        lcoa, _ = lcoa_levelized_lithium(pp)
        lcoas.append(lcoa)
    plt.figure()
    plt.plot(j_values, lcoas)
    plt.xlabel("Current density j_tot (A/m²)")
    plt.ylabel("LCOA (€/kg NH3)")
    plt.title("LCOA vs current density - Li mediated")
    plt.grid(True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parity_vs_price(params, eta_values, c_el_values, cB, output):
    """Plot parity current density vs electricity price for different ``eta`` values."""
    plt.figure()
    for eta in eta_values:
        j_req = [parity_j_for_eta_lithium(eta, p, params, cB) for p in c_el_values]
        plt.plot(c_el_values, j_req, label=f"η_FE={eta:.2f}")
    plt.xlabel("Electricity price (€/MWh)")
    plt.ylabel("j_tot for parity (A/m²)")
    plt.title("Parity current density vs electricity price")
    plt.legend()
    plt.grid(True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- Main execution ----------
def main():
    # Baseline parameters (illustrative only)
    params = {
        "eta_fe": 0.5,
        "j_tot": 1000.0,
        "U_cell": 3.0,
        "A": 10.0,
        "c_A": 5000.0,
        "c_P": 300.0,
        "c_el_MWh": 50.0,
        "f_OandM": 0.04,
        "i": 0.08,
        "N": 10,
        "t_op": 8000.0,
        "cA_BoP_extra": 0.0,
        "cP_BoP_extra": 0.0,
        "OPEX_BoP_extra_year": 0.0,
        "availability": 0.95,
        "deg_eta_yearly": 0.02,
        "deg_j_yearly": 0.02,
        "rep_interval_years": 5,
        "rep_frac_of_CAPEX_area": 0.2,
        "rep_frac_of_CAPEX_power": 0.2,
        "Li_loss_kg_per_kgNH3": 0.05,  # 50 g Li per kg NH3 lost
        "c_Li_per_kg": 100.0,          # €/kg Li
    }

    j_vals = np.linspace(100, 5000, 50)
    plot_lcoa_vs_j(params, j_vals, "lcoa_vs_j.png")

    eta_values = [0.3, 0.5, 0.7]
    c_el_values = np.linspace(20, 100, 50)
    plot_parity_vs_price(params, eta_values, c_el_values, cB=1.0, output="parity_vs_price.png")

    print("Generated lcoa_vs_j.png and parity_vs_price.png")


if __name__ == "__main__":
    main()
