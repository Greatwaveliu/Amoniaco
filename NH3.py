# -*- coding: utf-8 -*-
"""
GUI LCOA para NRR directa con:
- BoP, degradación, disponibilidad, reemplazos
- Sensibilidades 2D (LCOA)
- Mapa E_spec + EHB
- Paridad vs Precio de electricidad (curvas j_tot(η_FE) a paridad) con:
  * Banda "zona no factible"
  * Líneas horizontales j_min y j_max
  * Tabla y exportación a CSV (tabla y curvas completas)
"""

import sys, signal, csv, json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ---------- Constantes físico-químicas ----------
F = 96485.0      # C/mol
M_NH3 = 0.01703  # kg/mol
z = 6.0

# ---------- Utilidades de ingeniería económica ----------
def crf(i: float, N: int) -> float:
    """Capital Recovery Factor."""
    if i <= 0:
        return 1.0 / max(N, 1)
    return (i * (1 + i) ** N) / ((1 + i) ** N - 1)

def annual_worth_replacement(cost: float, i: float, k_years: int) -> float:
    """Equivalente anual de un reemplazo periódico de costo 'cost' cada k_years."""
    if k_years <= 0 or cost <= 0:
        return 0.0
    if i <= 0:
        return cost / k_years
    return cost * (i) / (1.0 - (1.0 + i) ** (-k_years))

# ---------- Modelos de degradación promedio ----------
def avg_factor_exponential(d: float, N: int) -> float:
    """
    Factor promedio (0-1) con decaimiento exponencial d (0-1) durante N años.
    E[X]/X0 = [(1 - (1 - d)^N) / (N*d)], d>0. Si d=0 => 1.0
    """
    if d <= 0:
        return 1.0
    return (1.0 - (1.0 - d) ** N) / (N * d)

# ---------- Núcleo de cálculo ----------
def production_annual(eta_fe, j_tot, A, t_op_h_per_y):
    """Producción anual (kg/año) por ley de Faraday."""
    mol_per_s = (eta_fe * j_tot * A) / (z * F)
    kg_per_s = mol_per_s * M_NH3
    return kg_per_s * 3600.0 * t_op_h_per_y

def e_specific_kwh_per_kg(U_cell, eta_fe):
    """E_spec [kWh/kg] = (zF/M) * (U/η) / 3.6e6 (vectorizable)."""
    K = (z * F / M_NH3) / 3.6e6
    return K * (U_cell / np.maximum(eta_fe, 1e-12))

def power_kw(j_tot, A, U_cell):
    """Potencia eléctrica [kW] ~ j*A*U / 1000"""
    return (j_tot * A * U_cell) / 1000.0

def lcoa_levelized(params):
    """
    LCOA con BoP, degradación, disponibilidad y reemplazos (equivalente anual).
    Retorna (LCOA, breakdown)
    """
    # Parámetros base
    eta0      = params["eta_fe"]
    j0        = params["j_tot"]
    U_cell    = params["U_cell"]
    A         = params["A"]
    c_A       = params["c_A"]
    c_P       = params["c_P"]
    c_el_MWh  = params["c_el_MWh"]
    f_OandM   = params["f_OandM"]
    i         = params["i"]
    N         = params["N"]
    t_op      = params["t_op"]

    # BoP (sumandos a c_A y c_P) + OPEX BoP directo
    cA_bop    = params["cA_BoP_extra"]
    cP_bop    = params["cP_BoP_extra"]
    opex_bop  = params["OPEX_BoP_extra_year"]  # €/año

    # Disponibilidad
    availability = params["availability"]
    t_op_eff = t_op * availability

    # Degradaciones (promedio N años)
    d_eta = params["deg_eta_yearly"]
    d_j   = params["deg_j_yearly"]
    f_eta_avg = avg_factor_exponential(d_eta, N)
    f_j_avg   = avg_factor_exponential(d_j, N)

    eta_eff = eta0 * f_eta_avg
    j_eff   = j0   * f_j_avg

    # CAPEX con BoP
    P_kw_eff = power_kw(j_eff, A, U_cell)
    CAPEX_area  = (c_A + cA_bop) * A
    CAPEX_power = (c_P + cP_bop) * P_kw_eff
    CAPEX_total = CAPEX_area + CAPEX_power

    # Reemplazos periódicos
    rep_every = params["rep_interval_years"]
    rep_fracA = params["rep_frac_of_CAPEX_area"]
    rep_fracP = params["rep_frac_of_CAPEX_power"]
    rep_cost = rep_fracA * CAPEX_area + rep_fracP * CAPEX_power
    AW_rep = annual_worth_replacement(rep_cost, i, rep_every)

    # CRF y OPEX
    CRF = crf(i, N)
    OPEX_OandM = f_OandM * CAPEX_total
    m_anual = production_annual(eta_eff, j_eff, A, t_op_eff)
    E_spec = e_specific_kwh_per_kg(U_cell, eta_eff)
    OPEX_el = E_spec * m_anual * (c_el_MWh / 1000.0)

    annual_cost = CRF * CAPEX_total + OPEX_OandM + OPEX_el + opex_bop + AW_rep
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
        "E_spec_kWh_per_kg": E_spec,
        "m_anual_kg": m_anual,
        "annual_cost_eur": annual_cost,
        "P_kw_eff": P_kw_eff,
        "eta_eff": eta_eff,
        "j_eff": j_eff,
        "t_op_eff": t_op_eff,
    }
    return lcoa, breakdown

# ---------- Paridad numérica: j requerido para LCOA=cB ----------
def parity_j_for_eta(eta, c_el_MWh, params, cB, j_lo=50.0, j_hi=50000.0, tol=1e-6, maxit=100):
    """
    Para una η dada y precio eléctrico c_el [€/MWh], encuentra j_tot que cumple LCOA=cB.
    Búsqueda por bisección (asumiendo LCOA decrece con j en el rango evaluado).
    Devuelve np.nan si no hay solución en [j_lo, j_hi].
    """
    pp = dict(params)
    pp["eta_fe"] = float(eta)
    pp["c_el_MWh"] = float(c_el_MWh)

    # LCOA en extremos
    pp["j_tot"] = j_lo
    f_lo, _ = lcoa_levelized(pp)
    pp["j_tot"] = j_hi
    f_hi, _ = lcoa_levelized(pp)

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
        f_mid, _ = lcoa_levelized(pp)
        f_mid -= cB
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) / (1.0 + mid) < tol:
            return hi
    return hi

# ---------- GUI ----------
class LCOAGUI:
    def __init__(self, root):
        self.root = root
        root.title("LCOA - Comparación de tecnologías")
        # Valores por defecto para cada tecnología (numéricos)
        direct_defaults = {
            "eta_fe": 0.60,
            "j_tot": 1000.0,
            "U_cell": 2.5,
            "A": 10.0,
            "c_A": 4298.9,
            "c_P": 240.5,
            "c_el_MWh": 30.0,
            "f_OandM": 0.04,
            "i": 0.08,
            "N": 10,
            "t_op": 8000.0,
            "cA_BoP_extra": 0.0,
            "cP_BoP_extra": 0.0,
            "OPEX_BoP_extra_year": 0.0,
            "availability": 0.95,
            "deg_eta_yearly": 0.0,
            "deg_j_yearly": 0.0,
            "rep_interval_years": 0,
            "rep_frac_of_CAPEX_area": 0.0,
            "rep_frac_of_CAPEX_power": 0.0,
            "c_B": 1.0,
        }
        li_defaults = dict(direct_defaults)
        # Diccionarios de parámetros por tecnología
        self.tech_defaults = {
            "NRR Directa": direct_defaults,
            "NRR Li-mediada": li_defaults,
        }
        self.tech_params = {k: dict(v) for k, v in self.tech_defaults.items()}
        self.tech_names = list(self.tech_defaults.keys())
        self.current_tech = tk.StringVar(value=self.tech_names[0])
        self.make_widgets()
        self._make_menu()

    def make_widgets(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Tecnología:").pack(side="left")
        self.cbo_tech = ttk.Combobox(top, values=self.tech_names, state="readonly",
                                     textvariable=self.current_tech)
        self.cbo_tech.pack(side="left", padx=4)
        self.cbo_tech.bind("<<ComboboxSelected>>", self.on_switch_tech)

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_inputs = ttk.Frame(nb)
        self.tab_sens   = ttk.Frame(nb)
        self.tab_espec  = ttk.Frame(nb)
        self.tab_price  = ttk.Frame(nb)
        self.tab_help   = ttk.Frame(nb)

        nb.add(self.tab_inputs, text="Parámetros y Cálculo")
        nb.add(self.tab_sens,   text="Sensibilidades 2D")
        nb.add(self.tab_espec,  text="Mapa E_spec + EHB")
        nb.add(self.tab_price,  text="Paridad vs Precio de electricidad")
        nb.add(self.tab_help,   text="Ayuda")

        # --- Tab: Parámetros y Cálculo ---
        frm = ttk.Frame(self.tab_inputs, padding=8); frm.pack(fill="both", expand=True)
        left  = ttk.Frame(frm); left.grid(row=0, column=0, sticky="n")
        mid   = ttk.Frame(frm);  mid.grid(row=0, column=1, sticky="n", padx=15)
        right = ttk.Frame(frm); right.grid(row=0, column=2, sticky="n")

        defaults = self.tech_defaults[self.tech_names[0]]
        # Entradas base
        self.entry_eta = self._add_entry(left, "η_FE base (0-1):", str(defaults["eta_fe"]))
        self.entry_j   = self._add_entry(left, "j_tot base (A/m²):", str(defaults["j_tot"]))
        self.entry_U   = self._add_entry(left, "U_cell (V):", str(defaults["U_cell"]))
        self.entry_A   = self._add_entry(left, "Área A (m²):", str(defaults["A"]))
        self.entry_cA  = self._add_entry(left, "c_A (€/m²):", str(defaults["c_A"]))
        self.entry_cP  = self._add_entry(left, "c_P (€/kW):", str(defaults["c_P"]))
        self.entry_cel = self._add_entry(left, "c_el (€/MWh):", str(defaults["c_el_MWh"]))
        self.entry_fO  = self._add_entry(left, "f_O&M (-):", str(defaults["f_OandM"]))
        self.entry_i   = self._add_entry(left, "i (tasa interés):", str(defaults["i"]))
        self.entry_N   = self._add_entry(left, "N (años):", str(defaults["N"]))
        self.entry_top = self._add_entry(left, "t_op (h/año):", str(defaults["t_op"]))

        # BoP + Disponibilidad
        ttk.Label(mid, text="Balance de Planta (BoP)").pack(anchor="w", pady=(6,2))
        self.entry_cA_bop = self._add_entry(mid, "c_A BoP extra (€/m²):", str(defaults["cA_BoP_extra"]))
        self.entry_cP_bop = self._add_entry(mid, "c_P BoP extra (€/kW):", str(defaults["cP_BoP_extra"]))
        self.entry_opex_bop = self._add_entry(mid, "OPEX BoP extra (€/año):", str(defaults["OPEX_BoP_extra_year"]))
        ttk.Label(mid, text="Disponibilidad y Degradación").pack(anchor="w", pady=(10,2))
        self.entry_avail = self._add_entry(mid, "Disponibilidad (0-1):", str(defaults["availability"]))
        self.entry_deg_eta = self._add_entry(mid, "Degradación η_FE (%/año):", str(defaults["deg_eta_yearly"]))
        self.entry_deg_j   = self._add_entry(mid, "Degradación j_tot (%/año):", str(defaults["deg_j_yearly"]))

        # Reemplazos y c_B
        ttk.Label(right, text="Reemplazos periódicos").pack(anchor="w", pady=(6,2))
        self.entry_rep_every = self._add_entry(right, "Cada k años:", str(defaults["rep_interval_years"]))
        self.entry_rep_fracA = self._add_entry(right, "Frac. CAPEX área a reemplazar:", str(defaults["rep_frac_of_CAPEX_area"]))
        self.entry_rep_fracP = self._add_entry(right, "Frac. CAPEX potencia a reemplazar:", str(defaults["rep_frac_of_CAPEX_power"]))
        ttk.Label(right, text="Precio de referencia (c_B)").pack(anchor="w", pady=(10,2))
        self.entry_cb = self._add_entry(right, "c_B (€/kg):", str(defaults["c_B"]))

        # Mapa de entradas para guardar/cargar parámetros fácilmente
        self.entry_map = {
            "eta_fe": self.entry_eta,
            "j_tot": self.entry_j,
            "U_cell": self.entry_U,
            "A": self.entry_A,
            "c_A": self.entry_cA,
            "c_P": self.entry_cP,
            "c_el_MWh": self.entry_cel,
            "f_OandM": self.entry_fO,
            "i": self.entry_i,
            "N": self.entry_N,
            "t_op": self.entry_top,
            "cA_BoP_extra": self.entry_cA_bop,
            "cP_BoP_extra": self.entry_cP_bop,
            "OPEX_BoP_extra_year": self.entry_opex_bop,
            "availability": self.entry_avail,
            "deg_eta_yearly": self.entry_deg_eta,
            "deg_j_yearly": self.entry_deg_j,
            "rep_interval_years": self.entry_rep_every,
            "rep_frac_of_CAPEX_area": self.entry_rep_fracA,
            "rep_frac_of_CAPEX_power": self.entry_rep_fracP,
            "c_B": self.entry_cb,
        }

        # Botones
        btns = ttk.Frame(self.tab_inputs, padding=8); btns.pack(fill="x")
        ttk.Button(btns, text="Calcular LCOA", command=self.on_calculate).pack(side="left", padx=4)
        ttk.Button(btns, text="Comparar tecnologías", command=self.on_compare).pack(side="left", padx=4)
        ttk.Button(btns, text="Mostrar desglose", command=self.on_breakdown).pack(side="left", padx=4)
        ttk.Button(btns, text="Comprobar unidades", command=self.on_units_check).pack(side="left", padx=4)
        ttk.Button(btns, text="Restablecer", command=self.reset_defaults).pack(side="left", padx=4)
        ttk.Button(btns, text="Salir", command=self.root.destroy).pack(side="right")

        self.lbl_res = ttk.Label(self.tab_inputs, text="LCOA = —", font=("Segoe UI", 12, "bold"))
        self.lbl_res.pack(pady=6)

        # Cargar parámetros iniciales de la tecnología seleccionada
        self.on_switch_tech()

        # --- Tab: Sensibilidades 2D (LCOA) ---
        fs = ttk.Frame(self.tab_sens, padding=8); fs.pack(fill="both", expand=True)
        choices = ["η_FE", "j_tot", "U_cell", "c_el"]
        ttk.Label(fs, text="Eje X").grid(row=0, column=0, sticky="w")
        self.cbo_x = ttk.Combobox(fs, values=choices, state="readonly"); self.cbo_x.set("j_tot")
        self.cbo_x.grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(fs, text="X mínimo").grid(row=0, column=2); self.ent_xmin = ttk.Entry(fs); self.ent_xmin.grid(row=0, column=3); self.ent_xmin.insert(0, "500")
        ttk.Label(fs, text="X máximo").grid(row=0, column=4); self.ent_xmax = ttk.Entry(fs); self.ent_xmax.grid(row=0, column=5); self.ent_xmax.insert(0, "2000")
        ttk.Label(fs, text="# puntos X").grid(row=0, column=6); self.ent_xn = ttk.Entry(fs, width=6); self.ent_xn.grid(row=0, column=7); self.ent_xn.insert(0, "20")

        ttk.Label(fs, text="Eje Y").grid(row=1, column=0, sticky="w", pady=(4,2))
        self.cbo_y = ttk.Combobox(fs, values=choices, state="readonly"); self.cbo_y.set("η_FE")
        self.cbo_y.grid(row=1, column=1, padx=6, pady=2)
        ttk.Label(fs, text="Y mínimo").grid(row=1, column=2); self.ent_ymin = ttk.Entry(fs); self.ent_ymin.grid(row=1, column=3); self.ent_ymin.insert(0, "0.3")
        ttk.Label(fs, text="Y máximo").grid(row=1, column=4); self.ent_ymax = ttk.Entry(fs); self.ent_ymax.grid(row=1, column=5); self.ent_ymax.insert(0, "0.9")
        ttk.Label(fs, text="# puntos Y").grid(row=1, column=6); self.ent_yn = ttk.Entry(fs, width=6); self.ent_yn.grid(row=1, column=7); self.ent_yn.insert(0, "20")

        ttk.Label(fs, text="Nivel de paridad (€/kg) c_B:").grid(row=2, column=0, sticky="w", pady=(8,2))
        self.ent_cb_plot = ttk.Entry(fs); self.ent_cb_plot.grid(row=2, column=1); self.ent_cb_plot.insert(0, "1.0")
        ttk.Button(fs, text="Generar mapa de paridad (contornos LCOA)", command=self.on_sensitivity).grid(row=2, column=2, columnspan=2, padx=6)

        # --- Tab: Mapa E_spec + EHB ---
        fe = ttk.Frame(self.tab_espec, padding=8); fe.pack(fill="both", expand=True)
        ttk.Label(fe, text="U_cell mínimo [V]").grid(row=0, column=0, sticky="w")
        self.es_u_min = ttk.Entry(fe); self.es_u_min.grid(row=0, column=1); self.es_u_min.insert(0, "1.5")
        ttk.Label(fe, text="U_cell máximo [V]").grid(row=0, column=2, sticky="w")
        self.es_u_max = ttk.Entry(fe); self.es_u_max.grid(row=0, column=3); self.es_u_max.insert(0, "3.5")
        ttk.Label(fe, text="η_FE mínimo [-]").grid(row=1, column=0, sticky="w", pady=(4,2))
        self.es_eta_min = ttk.Entry(fe); self.es_eta_min.grid(row=1, column=1); self.es_eta_min.insert(0, "0.10")
        ttk.Label(fe, text="η_FE máximo [-]").grid(row=1, column=2, sticky="w", pady=(4,2))
        self.es_eta_max = ttk.Entry(fe); self.es_eta_max.grid(row=1, column=3); self.es_eta_max.insert(0, "0.90")
        ttk.Label(fe, text="# puntos U").grid(row=2, column=0, sticky="w")
        self.es_nu = ttk.Entry(fe, width=8); self.es_nu.grid(row=2, column=1); self.es_nu.insert(0, "200")
        ttk.Label(fe, text="# puntos η").grid(row=2, column=2, sticky="w")
        self.es_neta = ttk.Entry(fe, width=8); self.es_neta.grid(row=2, column=3); self.es_neta.insert(0, "200")
        ttk.Label(fe, text="EHB bajo [kWh/kg]").grid(row=3, column=0, sticky="w", pady=(6,2))
        self.es_ehb_low = ttk.Entry(fe); self.es_ehb_low.grid(row=3, column=1); self.es_ehb_low.insert(0, "10.5")
        ttk.Label(fe, text="EHB alto [kWh/kg]").grid(row=3, column=2, sticky="w", pady=(6,2))
        self.es_ehb_high = ttk.Entry(fe); self.es_ehb_high.grid(row=3, column=3); self.es_ehb_high.insert(0, "11.0")
        ttk.Label(fe, text="Guardar PNG (opcional)").grid(row=4, column=0, sticky="w", pady=(8,2))
        self.es_save = ttk.Entry(fe, width=40); self.es_save.grid(row=4, column=1, columnspan=3, sticky="we")
        ttk.Button(fe, text="Generar mapa E_spec + EHB", command=self.on_espec_map).grid(row=5, column=0, columnspan=4, pady=10)

        # --- Tab: Paridad vs Precio de electricidad ---
        fp = ttk.Frame(self.tab_price, padding=8); fp.pack(fill="both", expand=True)
        ttk.Label(fp, text="Lista precios electricidad [€/MWh] (coma)").grid(row=0, column=0, sticky="w")
        self.prices_entry = ttk.Entry(fp, width=50); self.prices_entry.grid(row=0, column=1, sticky="we"); self.prices_entry.insert(0, "20,30,40,60,80")
        ttk.Label(fp, text="Rango η_FE: min").grid(row=1, column=0, sticky="w", pady=(6,2))
        self.p_eta_min = ttk.Entry(fp, width=10); self.p_eta_min.grid(row=1, column=1, sticky="w"); self.p_eta_min.insert(0, "0.10")
        ttk.Label(fp, text="max").grid(row=1, column=1, sticky="e")
        self.p_eta_max = ttk.Entry(fp, width=10); self.p_eta_max.grid(row=1, column=1, padx=(60,0)); self.p_eta_max.insert(0, "0.90")
        ttk.Label(fp, text="# puntos η").grid(row=1, column=1, sticky="e", padx=(140,0))
        self.p_eta_n = ttk.Entry(fp, width=8); self.p_eta_n.grid(row=1, column=1, sticky="e", padx=(230,0)); self.p_eta_n.insert(0, "40")
        ttk.Label(fp, text="c_B (€/kg)").grid(row=2, column=0, sticky="w", pady=(6,2))
        self.p_cB = ttk.Entry(fp, width=10); self.p_cB.grid(row=2, column=1, sticky="w"); self.p_cB.insert(0, "1.0")
        ttk.Label(fp, text="U_cell [V]").grid(row=3, column=0, sticky="w")
        self.p_U = ttk.Entry(fp, width=10); self.p_U.grid(row=3, column=1, sticky="w"); self.p_U.insert(0, "2.5")
        ttk.Label(fp, text="Límites j_tot [A/m²]: min").grid(row=4, column=0, sticky="w", pady=(6,2))
        self.p_jmin = ttk.Entry(fp, width=10); self.p_jmin.grid(row=4, column=1, sticky="w"); self.p_jmin.insert(0, "50")
        ttk.Label(fp, text="max").grid(row=4, column=1, sticky="e")
        self.p_jmax = ttk.Entry(fp, width=10); self.p_jmax.grid(row=4, column=1, padx=(60,0)); self.p_jmax.insert(0, "50000")
        ttk.Label(fp, text="Guardar PNG (opcional)").grid(row=5, column=0, sticky="w", pady=(6,2))
        self.p_save = ttk.Entry(fp, width=40); self.p_save.grid(row=5, column=1, sticky="we")
        ttk.Button(fp, text="Graficar curvas de paridad vs precio", command=self.on_price_parity).grid(row=6, column=0, columnspan=2, pady=10)

        # --- Tab: Ayuda ---
        txt = tk.Text(self.tab_help, wrap="word", height=18)
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("end",
            "Guía rápida:\n"
            "• 'Sensibilidades 2D': mapas de LCOA en dos variables (ej. j_tot vs η_FE).\n"
            "• 'Mapa E_spec + EHB': energía específica vs U_cell y η_FE con banda EHB (10.5–11 kWh/kg).\n"
            "• 'Paridad vs Precio de electricidad': curvas j_tot(η_FE) a LCOA=c_B y U_cell fijo; \n"
            "   incluye zona no factible, líneas j_min/j_max, tabla y exportación CSV.\n"
        )
        txt.configure(state="disabled")

    def _add_entry(self, parent, label, default=""):
        row = ttk.Frame(parent); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
        ent = ttk.Entry(row, width=16); ent.insert(0, default); ent.pack(side="left")
        return ent

    def _read_params(self):
        try:
            params = {
                "eta_fe":      float(self.entry_eta.get()),
                "j_tot":       float(self.entry_j.get()),
                "U_cell":      float(self.entry_U.get()),
                "A":           float(self.entry_A.get()),
                "c_A":         float(self.entry_cA.get()),
                "c_P":         float(self.entry_cP.get()),
                "c_el_MWh":    float(self.entry_cel.get()),
                "f_OandM":     float(self.entry_fO.get()),
                "i":           float(self.entry_i.get()),
                "N":           int(float(self.entry_N.get())),
                "t_op":        float(self.entry_top.get()),
                "cA_BoP_extra": float(self.entry_cA_bop.get()),
                "cP_BoP_extra": float(self.entry_cP_bop.get()),
                "OPEX_BoP_extra_year": float(self.entry_opex_bop.get()),
                "availability": float(self.entry_avail.get()),
                "deg_eta_yearly": float(self.entry_deg_eta.get())/100.0,
                "deg_j_yearly":   float(self.entry_deg_j.get())/100.0,
                "rep_interval_years": int(float(self.entry_rep_every.get())),
                "rep_frac_of_CAPEX_area":  float(self.entry_rep_fracA.get()),
                "rep_frac_of_CAPEX_power": float(self.entry_rep_fracP.get()),
                "c_B": float(self.entry_cb.get())
            }
        except Exception as e:
            raise ValueError(f"Entrada inválida: {e}")
        if not (0 < params["eta_fe"] <= 1):
            raise ValueError("η_FE debe estar en (0,1].")
        if not (0 < params["availability"] <= 1):
            raise ValueError("Disponibilidad debe estar en (0,1].")
        if params["N"] <= 0:
            raise ValueError("N (años) debe ser > 0.")
        self.tech_params[self.current_tech.get()] = dict(params)
        return params

    # ------ Gestión de parámetros: cargar/guardar/restablecer ------
    def _make_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Cargar parámetros…", command=self.load_params)
        filemenu.add_command(label="Guardar parámetros…", command=self.save_params)
        filemenu.add_command(label="Restablecer valores", command=self.reset_defaults)
        filemenu.add_separator()
        filemenu.add_command(label="Salir", command=self.root.destroy)
        menubar.add_cascade(label="Archivo", menu=filemenu)
        self.root.config(menu=menubar)

    def reset_defaults(self):
        params = self.tech_defaults[self.current_tech.get()]
        for key, entry in self.entry_map.items():
            entry.delete(0, tk.END)
            if key in params:
                entry.insert(0, str(params[key]))
        self.lbl_res.config(text="LCOA = —")

    def save_params(self):
        path = filedialog.asksaveasfilename(title="Guardar parámetros", defaultextension=".json",
                                            filetypes=[("JSON", "*.json"), ("Todos", "*.*")])
        if not path:
            return
        try:
            params = self._read_params()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")

    def load_params(self):
        path = filedialog.askopenfilename(title="Cargar parámetros", defaultextension=".json",
                                          filetypes=[("JSON", "*.json"), ("Todos", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer: {e}")
            return
        for key, val in data.items():
            if key in self.entry_map:
                ent = self.entry_map[key]
                ent.delete(0, tk.END)
                ent.insert(0, str(val))
        try:
            self._read_params()
        except Exception:
            pass

    def on_switch_tech(self, event=None):
        tech = self.current_tech.get()
        params = self.tech_params.get(tech, self.tech_defaults[tech])
        for key, entry in self.entry_map.items():
            entry.delete(0, tk.END)
            if key in params:
                entry.insert(0, str(params[key]))
        self.lbl_res.config(text="LCOA = —")

    def on_compare(self):
        try:
            current = self._read_params()
            self.tech_params[self.current_tech.get()] = dict(current)
            lines = []
            for tech in self.tech_names:
                val, _ = lcoa_levelized(self.tech_params[tech])
                lines.append(f"{tech}: {val:.4f} €/kg")
            messagebox.showinfo("Comparación de tecnologías", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------ Acciones principales ------
    def on_calculate(self):
        try:
            p = self._read_params()
            lcoa, _ = lcoa_levelized(p)
            self.lbl_res.config(text=f"LCOA = {lcoa:.3f} €/kg NH₃")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_breakdown(self):
        try:
            p = self._read_params()
            lcoa, b = lcoa_levelized(p)
            msg = (
                f"LCOA = {lcoa:.4f} €/kg\n\n"
                f"CRF = {b['CRF']:.5f}\n"
                f"CAPEX área = {b['CAPEX_area']:.2f} €\n"
                f"CAPEX potencia = {b['CAPEX_power']:.2f} €\n"
                f"CAPEX total = {b['CAPEX_total']:.2f} €\n"
                f"OPEX O&M = {b['OPEX_OandM']:.2f} € /año\n"
                f"OPEX eléctrico = {b['OPEX_el']:.2f} € /año\n"
                f"OPEX BoP extra = {b['OPEX_BoP_extra_year']:.2f} € /año\n"
                f"AW reemplazos = {b['AW_replacement']:.2f} € /año\n"
                f"E_spec = {b['E_spec_kWh_per_kg']:.3f} kWh/kg\n"
                f"Producción anual = {b['m_anual_kg']:.2f} kg/año\n"
                f"Potencia efectiva = {b['P_kw_eff']:.3f} kW\n"
                f"η_FE efectiva = {b['eta_eff']:.4f}\n"
                f"j_tot efectivo = {b['j_eff']:.2f} A/m²\n"
                f"t_op efectivo = {b['t_op_eff']:.1f} h/año\n"
                f"Coste anual total = {b['annual_cost_eur']:.2f} € /año\n"
            )
            messagebox.showinfo("Desglose", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_units_check(self):
        try:
            p = self._read_params()
            _, b = lcoa_levelized(p)
            c_el_per_kWh = p["c_el_MWh"] / 1000.0
            lhs = b["E_spec_kWh_per_kg"] * b["m_anual_kg"] * c_el_per_kWh
            msg = (
                "Chequeo de unidad:\n"
                "E_spec[kWh/kg] × m_anual[kg/año] × c_el[€/kWh] → €/año\n\n"
                f"Resultado: {lhs:.2f} €/año\n"
                f"OPEX_el (modelo) = {b['OPEX_el']:.2f} €/año\n"
            )
            messagebox.showinfo("Chequeo de unidades", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_sensitivity(self):
        try:
            p = self._read_params()
            xvar = self.cbo_x.get()
            yvar = self.cbo_y.get()
            if xvar == yvar:
                raise ValueError("Elige variables distintas para X e Y.")

            x_min = float(self.ent_xmin.get()); x_max = float(self.ent_xmax.get()); xn = int(self.ent_xn.get())
            y_min = float(self.ent_ymin.get()); y_max = float(self.ent_ymax.get()); yn = int(self.ent_yn.get())
            cB = float(self.ent_cb_plot.get())

            X = np.linspace(x_min, x_max, max(xn, 2))
            Y = np.linspace(y_min, y_max, max(yn, 2))
            XX, YY = np.meshgrid(X, Y)
            ZZ = np.zeros_like(XX, dtype=float)

            def apply_xy(pp, val, name):
                if name == "η_FE":
                    pp["eta_fe"] = float(val)
                elif name == "j_tot":
                    pp["j_tot"] = float(val)
                elif name == "U_cell":
                    pp["U_cell"] = float(val)
                elif name == "c_el":
                    pp["c_el_MWh"] = float(val)
                else:
                    raise ValueError(f"Variable desconocida: {name}")

            for i in range(YY.shape[0]):
                for j in range(YY.shape[1]):
                    pp = dict(p)
                    apply_xy(pp, XX[i, j], xvar)
                    apply_xy(pp, YY[i, j], yvar)
                    val, _ = lcoa_levelized(pp)
                    ZZ[i, j] = val

            fig, ax = plt.subplots()
            cs = ax.contourf(XX, YY, ZZ, levels=20)
            fig.colorbar(cs, ax=ax, label="LCOA [€/kg NH₃]")
            try:
                cset = ax.contour(XX, YY, ZZ, levels=[cB], linewidths=2)
                ax.clabel(cset, fmt={cB: f"Paridad {cB:.2f} €/kg"})
            except Exception:
                pass
            ax.set_xlabel(xvar); ax.set_ylabel(yvar); ax.set_title("Mapa de paridad LCOA (contornos)")
            fig.tight_layout(rect=[0, 0.07, 1, 1])
            btn_ax = fig.add_axes([0.55, 0.01, 0.4, 0.05])
            self._sens_btn = Button(btn_ax, "Guardar Datos de Curvas")
            self._sens_btn.on_clicked(self.on_sens_save)
            self._sens_last = {
                "tech": self.current_tech.get(),
                "XX": XX,
                "YY": YY,
                "ZZ": ZZ,
                "xvar": xvar,
                "yvar": yvar,
                "cB": cB,
            }
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_sens_save(self, event=None):
        try:
            data = getattr(self, "_sens_last", None)
            if not data:
                messagebox.showinfo("Sin datos", "No hay curvas generadas para guardar.")
                return

            XX = data["XX"]; YY = data["YY"]; ZZ = data["ZZ"]
            xvar = data["xvar"]; yvar = data["yvar"]; cB = data["cB"]

            rows = [(xvar, yvar)]
            fig = plt.figure()
            try:
                cs = plt.contour(XX, YY, ZZ, levels=[cB])
                for path in cs.collections[0].get_paths():
                    verts = path.vertices
                    for vx, vy in verts:
                        rows.append((f"{vx:.6f}", f"{vy:.6f}"))
            except Exception:
                pass
            finally:
                plt.close(fig)

            if len(rows) <= 1:
                messagebox.showinfo("Sin datos", "No se encontraron curvas de paridad en el rango especificado.")
                return

            path = filedialog.asksaveasfilename(
                title="Guardar datos de curvas como CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
            )
            if not path:
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in rows:
                    w.writerow(r)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_espec_map(self):
        try:
            u_min   = float(self.es_u_min.get()); u_max   = float(self.es_u_max.get())
            eta_min = float(self.es_eta_min.get()); eta_max = float(self.es_eta_max.get())
            n_u     = int(self.es_nu.get()); n_eta   = int(self.es_neta.get())
            ehb_low  = float(self.es_ehb_low.get()); ehb_high = float(self.es_ehb_high.get())
            save_png = self.es_save.get().strip() or None

            if not (0 < eta_min < eta_max <= 1.0): raise ValueError("η_FE en (0,1], eta_min < eta_max.")
            if not (u_min < u_max): raise ValueError("U_cell: u_min < u_max.")

            U = np.linspace(u_min, u_max, max(n_u, 2))
            ETA = np.linspace(eta_min, eta_max, max(n_eta, 2))
            UU, EE = np.meshgrid(U, ETA)
            E_spec = e_specific_kwh_per_kg(UU, EE)

            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            cf = ax.contourf(UU, EE, E_spec, levels=25)
            fig.colorbar(cf, ax=ax, label="Energía específica [kWh/kg NH₃]")
            cs = ax.contour(UU, EE, E_spec, levels=[ehb_low, ehb_high], linewidths=2)
            ax.clabel(cs, fmt={ehb_low: f"{ehb_low} kWh/kg", ehb_high: f"{ehb_high} kWh/kg"})
            ax.set_xlabel("U_cell [V]"); ax.set_ylabel("η_FE [–]")
            ax.set_title("Demanda específica de energía del NRR con banda EHB (10.5–11 kWh/kg)")
            plt.tight_layout()
            if save_png: fig.savefig(save_png, dpi=200)
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------- Paridad vs Precio de electricidad --------
    def on_price_parity(self):
        try:
            # Parámetros base y forzamos U_cell y c_B desde la pestaña
            p = self._read_params()
            cB = float(self.p_cB.get())
            U_fix = float(self.p_U.get())
            p["U_cell"] = U_fix  # p.ej., 2.5 V

            # Rango de eficiencias para curvas
            eta_min = float(self.p_eta_min.get())
            eta_max = float(self.p_eta_max.get())
            n_eta   = int(self.p_eta_n.get())
            if not (0 < eta_min < eta_max <= 1.0):
                raise ValueError("η_FE en (0,1], con eta_min < eta_max.")
            etas = np.linspace(eta_min, eta_max, max(n_eta, 2))

            # Precios de electricidad (lista separada por coma o punto y coma)
            prices = [float(s) for s in self.prices_entry.get().replace(";", ",").split(",") if s.strip()]

            # Límites de búsqueda para j_tot
            jmin = float(self.p_jmin.get())
            jmax = float(self.p_jmax.get())
            if not (jmin < jmax):
                raise ValueError("Rango de j_tot inválido.")

            save_png = self.p_save.get().strip() or None

            # Curvas de paridad y máscara de factibilidad (al menos un precio tiene solución)
            feasible_any = np.zeros_like(etas, dtype=bool)
            curves_dict = {}  # guardaremos las curvas completas para exportar

            plt.figure()
            for c_el in prices:
                j_req = []
                for eta in etas:
                    j_star = parity_j_for_eta(eta, c_el, p, cB, j_lo=jmin, j_hi=jmax)
                    j_req.append(j_star)
                j_req = np.array(j_req, dtype=float)
                curves_dict[c_el] = j_req

                mask = ~np.isnan(j_req)
                feasible_any |= mask

                if mask.any():
                    plt.plot(etas[mask], j_req[mask], label=f"{c_el:.0f} €/MWh")

            # Líneas horizontales en jmin y jmax (rango de búsqueda)
            plt.axhline(jmin, linestyle="--", alpha=0.6, linewidth=1.2, label=f"j_min={jmin:.0f}")
            plt.axhline(jmax, linestyle="--", alpha=0.6, linewidth=1.2, label=f"j_max={jmax:.0f}")

            # Banda vertical de "zona no factible" (ningún precio logra paridad en [jmin, jmax])
            def spans_from_mask(x, ok_mask):
                spans = []
                in_gap = False
                start = None
                for k, ok in enumerate(ok_mask):
                    if (not ok) and (not in_gap):
                        start = x[k]; in_gap = True
                    elif ok and in_gap:
                        spans.append((start, x[k])); in_gap = False
                if in_gap:
                    spans.append((start, x[-1]))
                return spans

            not_feasible_spans = spans_from_mask(etas, feasible_any)
            added_label = False
            for a, b in not_feasible_spans:
                plt.axvspan(a, b, alpha=0.15,
                            label="Zona no factible (sin paridad en [j_min, j_max])" if not added_label else None)
                added_label = True

            # Decoración
            plt.xlabel("η_FE [–]")
            plt.ylabel("j_tot requerido para paridad [A/m²]")
            plt.title(
                f"Curvas de paridad LCOA=c_B={cB:.2f} €/kg a U_cell={U_fix:.2f} V\n"
                "Influencia del precio de electricidad (mayor efecto a baja η_FE)"
            )
            plt.ylim(jmin, jmax)
            plt.legend(title="Precio electricidad", loc="best")
            plt.tight_layout()
            if save_png:
                plt.gcf().savefig(save_png, dpi=200)
            plt.show(block=False)

            # -------- Tabla con j_tot a paridad en etas fijas + botones CSV --------
            etas_table = [0.2, 0.4, 0.6, 0.8]
            etas_table = [e for e in etas_table if 0 < e <= 1.0]

            # Calcula j* para cada precio y cada eta de la tabla
            table_vals = []
            for c_el in prices:
                row = []
                for eta in etas_table:
                    j_star = parity_j_for_eta(eta, c_el, p, cB, j_lo=jmin, j_hi=jmax)
                    row.append(np.nan if (j_star is None) else j_star)
                table_vals.append(row)

            # Ventana de tabla
            win = tk.Toplevel(self.root)
            win.title("Tabla j_tot requerido a paridad (LCOA=c_B)")
            cols = ["Precio [€/MWh]"] + [f"η_FE={e:.1f}" for e in etas_table]
            tree = ttk.Treeview(win, columns=cols, show="headings", height=min(12, len(prices)+2))
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, width=130, anchor="center")
            for i, c_el in enumerate(prices):
                vals = [f"{c_el:.0f}"]
                for val in table_vals[i]:
                    vals.append("—" if (val is None or np.isnan(val)) else f"{val:.0f}")
                tree.insert("", "end", values=vals)
            tree.pack(fill="both", expand=True, padx=8, pady=8)

            # Botonera inferior: Guardar CSV (tabla), Guardar CSV (curvas), Cerrar
            btn_row = ttk.Frame(win); btn_row.pack(fill="x", padx=8, pady=(0,8))

            def _save_csv_table():
                path = filedialog.asksaveasfilename(
                    title="Guardar tabla de paridad como CSV",
                    defaultextension=".csv",
                    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
                )
                if not path:
                    return
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(cols)
                    for i, c_el in enumerate(prices):
                        row = [f"{c_el:.0f}"]
                        for val in table_vals[i]:
                            row.append("NA" if (val is None or np.isnan(val)) else f"{val:.0f}")
                        w.writerow(row)

            def _save_csv_curves():
                path = filedialog.asksaveasfilename(
                    title="Guardar curvas completas como CSV",
                    defaultextension=".csv",
                    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
                )
                if not path:
                    return
                # Formato ancho: eta_FE, j_at_<P1>, j_at_<P2>, ...
                header = ["eta_FE"] + [f"j_tot_at_{int(p)}_EUR_per_MWh" for p in prices]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    for idx, eta in enumerate(etas):
                        row = [f"{eta:.6f}"]
                        for p_el in prices:
                            val = curves_dict[p_el][idx]
                            row.append("NA" if np.isnan(val) else f"{val:.6f}")
                        w.writerow(row)

            ttk.Button(btn_row, text="Guardar CSV (tabla)…", command=_save_csv_table).pack(side="left")
            ttk.Button(btn_row, text="Guardar CSV (curvas)…", command=_save_csv_curves).pack(side="left", padx=6)
            ttk.Button(btn_row, text="Cerrar", command=win.destroy).pack(side="right")

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- Main ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = LCOAGUI(root)

    # Manejo limpio de Ctrl+C en consola
    def _sigint_handler(sig, frame):
        try:
            root.quit(); root.destroy()
        except Exception:
            pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass

    try:
        root.mainloop()
    except KeyboardInterrupt:
        _sigint_handler(None, None)
