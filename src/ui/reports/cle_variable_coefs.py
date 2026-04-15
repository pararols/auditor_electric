"""
CLE Pavelló — Optimitzador de Coeficients Variables Horaris
============================================================
Calcula una matriu de coeficients (12 mesos × 24 hores) per a cada CUPS
municipal, repartint la quota del Pavelló proporcionalment al consum real
de cada franja horària. En nits d'estiu (Jun-Ago, 21h-07h) redistribueix
quota cap als CUPS d'enllumeament per evitar excedents no compensables.

Els coeficients de la Sala Nova (COMMUNITY_QUOTAS) no es toquen mai.
Normativa: RD 244/2019 — coeficients dinàmics basats en criteri objectiu.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.core.config import CUPS_MAPPING, COMMUNITY_QUOTAS

# Constants (igual que cle_optimizer.py)
PAVELLO_KWP = 127.2
TARGET_RATIO = 78.198 / PAVELLO_KWP  # ≈ 0.6148

MESOS_CAT = ['Gen', 'Feb', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Oct', 'Nov', 'Des']

SUMMER_MONTHS = {6, 7, 8}
NIGHT_HOURS = set(range(0, 7)) | {21, 22, 23}


def _is_lighting(cups_id: str) -> bool:
    """Retorna True si el CUPS és un quadre d'enllumeament públic."""
    name = CUPS_MAPPING.get(cups_id, "")
    return (
        name.startswith("Enll") or
        "Diana" in name or
        "repòs" in name.lower() or
        "Rotonda" in name
    )


def compute_variable_coef_matrix(
    df_consum: pd.DataFrame,
    summer_night_boost: float = 0.20,
    min_kwp_threshold: float = 0.5,
) -> dict:
    """
    Calcula la matriu de coeficients variables del Pavelló.

    Parameters
    ----------
    df_consum : DataFrame
        Consum horari per CUPS (columnes = cups_ids, índex DatetimeIndex).
    summer_night_boost : float
        Fracció (0–0.5) de la quota dels edificis que es redistribueix
        als CUPS d'enllumeament en nits d'estiu.
    min_kwp_threshold : float
        CUPS amb assignació mitjana anual < X kWp s'exclouen (igual que SLSQP).

    Returns
    -------
    dict[cups_id] -> np.ndarray(12, 24)
        Coeficients arrodonits a 6 decimals. Per a cada (m, h):
        sum(coefs[:, m, h]) == TARGET_RATIO.
    """
    cups_ids = list(df_consum.columns)
    N = len(cups_ids)

    months_v = df_consum.index.month.values
    hours_v = df_consum.index.hour.values
    lighting_mask = np.array([_is_lighting(c) for c in cups_ids], dtype=bool)
    building_mask = ~lighting_mask

    # --- Pas 1: consum mig per (CUPS, mes, hora) ---
    mean_cons = np.zeros((N, 12, 24))
    for i, cups_id in enumerate(cups_ids):
        vals = df_consum[cups_id].values
        for m_idx in range(12):
            for h in range(24):
                mask = (months_v == m_idx + 1) & (hours_v == h)
                if mask.any():
                    mean_cons[i, m_idx, h] = vals[mask].mean()

    # --- Pas 2: repartiment proporcional base ---
    raw_coef = np.zeros((N, 12, 24))
    for m_idx in range(12):
        for h in range(24):
            demands = mean_cons[:, m_idx, h]
            total_d = demands.sum()
            if total_d > 0:
                raw_coef[:, m_idx, h] = (demands / total_d) * TARGET_RATIO
            else:
                raw_coef[:, m_idx, h] = TARGET_RATIO / N

    # --- Pas 3: boost nocturn d'estiu cap als enllumeaments ---
    if summer_night_boost > 0 and lighting_mask.any() and building_mask.any():
        for m_idx in range(12):
            month_num = m_idx + 1
            if month_num not in SUMMER_MONTHS:
                continue
            for h in range(24):
                if h not in NIGHT_HOURS:
                    continue
                demands = mean_cons[:, m_idx, h]

                bld_coef_sum = raw_coef[building_mask, m_idx, h].sum()
                boost = bld_coef_sum * summer_night_boost

                # Repartir boost als enllumeaments proporcional al seu consum
                light_d = demands[lighting_mask]
                light_total = light_d.sum()
                if light_total > 0:
                    raw_coef[lighting_mask, m_idx, h] += boost * (light_d / light_total)

                    # Restar proporcionalment dels edificis
                    bld_d = demands[building_mask]
                    bld_total = bld_d.sum()
                    if bld_total > 0:
                        raw_coef[building_mask, m_idx, h] -= boost * (bld_d / bld_total)
                        raw_coef[building_mask, m_idx, h] = np.maximum(
                            0.0, raw_coef[building_mask, m_idx, h]
                        )

                # Re-normalitzar per garantir la suma exacta
                col_sum = raw_coef[:, m_idx, h].sum()
                if col_sum > 0:
                    raw_coef[:, m_idx, h] *= TARGET_RATIO / col_sum

    # --- Pas 4: excloure CUPS per sota del llindar kWp ---
    mean_annual_coef = raw_coef.mean(axis=(1, 2))
    excluded = (mean_annual_coef * PAVELLO_KWP) < min_kwp_threshold

    if excluded.any():
        raw_coef[excluded] = 0.0
        active = ~excluded
        for m_idx in range(12):
            for h in range(24):
                deficit = TARGET_RATIO - raw_coef[:, m_idx, h].sum()
                if deficit < 1e-9:
                    continue
                active_sum = raw_coef[active, m_idx, h].sum()
                if active_sum > 0:
                    raw_coef[active, m_idx, h] += deficit * (
                        raw_coef[active, m_idx, h] / active_sum
                    )
                elif active.sum() > 0:
                    raw_coef[active, m_idx, h] = deficit / active.sum()

    # --- Pas 5: arrodoniment legal a 6 decimals (suma = TARGET_RATIO) ---
    target_int = int(round(TARGET_RATIO * 1e6))
    final_coef = np.zeros((N, 12, 24))

    for m_idx in range(12):
        for h in range(24):
            c = raw_coef[:, m_idx, h]
            c_int = np.floor(c * 1e6).astype(np.int64)
            remainders = (c * 1e6) - c_int
            diff = target_int - int(c_int.sum())
            if diff > 0:
                top_idx = np.argsort(remainders)[-diff:]
                c_int[top_idx] += 1
            final_coef[:, m_idx, h] = c_int / 1e6

    return {cups_ids[i]: final_coef[i] for i in range(N)}


def evaluate_variable_coefs(
    coef_matrix: dict,
    cups_names: list,
    df_consum: pd.DataFrame,
    gen_pavello: np.ndarray,
    gen_salanova: np.ndarray,
    prices: np.ndarray,
    excedent_price: float,
) -> tuple:
    """
    Simula la facturació anual neta amb coeficients variables hora per hora.
    Aplica el topall mensual RD 244/2019 i els impostos (IE 5.11% + IVA 21%).

    Returns
    -------
    (results_by_cups: list[dict], total_savings: float)
    """
    months_v = df_consum.index.month.values
    hours_v = df_consum.index.hour.values

    total_cost_nosolar = 0.0
    total_cost_solar = 0.0
    results_by_cups = []

    for cups in cups_names:
        consum = df_consum[cups].values

        # Coeficient Sala Nova fix (de COMMUNITY_QUOTAS)
        sn_info = COMMUNITY_QUOTAS.get(cups)
        coef_sn = sn_info['coef'] if sn_info else 0.0

        # Coeficient Pavelló variable: vector horari
        if cups in coef_matrix:
            mat = coef_matrix[cups]  # (12, 24)
            coef_pav_vec = mat[months_v - 1, hours_v]
        else:
            coef_pav_vec = np.zeros(len(df_consum))

        gen_sn = gen_salanova * coef_sn
        gen_pav = gen_pavello * coef_pav_vec
        gen_total = gen_sn + gen_pav

        autoconsum = np.minimum(consum, gen_total)
        net_import = consum - autoconsum
        excedents = gen_total - autoconsum

        cost_nosolar_cups = 0.0
        cost_solar_cups = 0.0
        estalvi_auto_cups = 0.0
        estalvi_comp_cups = 0.0
        excedents_compensats = 0.0
        excedents_abocats = 0.0
        mensual_stats = []

        for m in range(1, 13):
            mask = (months_v == m)

            c_nos = float(np.sum(consum[mask] * prices[mask]))
            c_nos_taxes = c_nos * 1.0511 * 1.21
            cost_nosolar_cups += c_nos_taxes

            c_sol_import = float(np.sum(net_import[mask] * prices[mask]))
            v_exc = float(np.sum(excedents[mask])) * excedent_price
            terme_net = max(0.0, c_sol_import - v_exc)
            c_sol_taxes = terme_net * 1.0511 * 1.21
            cost_solar_cups += c_sol_taxes

            estalvi_auto_cups += (c_nos - c_sol_import) * 1.0511 * 1.21
            estalvi_comp_cups += (c_sol_import - terme_net) * 1.0511 * 1.21

            exc_total_m = float(np.sum(excedents[mask]))
            if v_exc <= c_sol_import:
                exc_comp = exc_total_m
                exc_lost = 0.0
            else:
                exc_comp = c_sol_import / excedent_price if excedent_price > 0 else 0.0
                exc_lost = exc_total_m - exc_comp

            excedents_compensats += exc_comp
            excedents_abocats += exc_lost

            # Proporcions per planta
            gen_t_m = gen_total[mask]
            gen_sn_m = gen_sn[mask]
            gen_pav_m = gen_pav[mask]
            auto_m = autoconsum[mask]
            valid = gen_t_m > 0
            denom = np.where(valid, gen_t_m, 1.0)
            prop_sn = np.where(valid, gen_sn_m / denom, 0.0)
            prop_pav = np.where(valid, gen_pav_m / denom, 0.0)

            mensual_stats.append({
                'Mes': m,
                'Consum': float(np.sum(consum[mask])),
                'Generació Assignada': float(np.sum(gen_t_m)),
                'Generació Assignada SN': float(np.sum(gen_sn_m)),
                'Generació Assignada PAV': float(np.sum(gen_pav_m)),
                'Autoconsum SN': float(np.sum(auto_m * prop_sn)),
                'Autoconsum PAV': float(np.sum(auto_m * prop_pav)),
                'Autoconsum': float(np.sum(auto_m)),
                'Import Net': float(np.sum(net_import[mask])),
                'Estalvi € (Brut)': c_nos_taxes - c_sol_taxes,
            })

        mean_coef_pav = float(coef_pav_vec.mean())
        total_auto_sn = sum(x['Autoconsum SN'] for x in mensual_stats)
        total_auto_pav = sum(x['Autoconsum PAV'] for x in mensual_stats)
        total_auto = float(np.sum(autoconsum))
        total_cons = float(np.sum(consum))

        results_by_cups.append({
            'CUPS': cups,
            'Nom': CUPS_MAPPING.get(cups, ''),
            'Coeficient Sala Nova': coef_sn,
            'Potència Sala Nova (kWp)': coef_sn * 17.1,
            'Coeficient Pavelló (Mig)': mean_coef_pav,
            'Potència Pavelló Mig (kWp)': mean_coef_pav * PAVELLO_KWP,
            'Consum Anual (kWh)': total_cons,
            'Producció FV (kWh)': sum(x['Generació Assignada'] for x in mensual_stats),
            'Autoconsum SN (kWh)': total_auto_sn,
            'Autoconsum PAV (kWh)': total_auto_pav,
            'Autoconsum Total (kWh)': total_auto,
            'Cobertura (%)': (total_auto / total_cons * 100) if total_cons > 0 else 0.0,
            'Excedents Compensats (kWh)': excedents_compensats,
            'Excedents Llençats a la xarxa (kWh)': excedents_abocats,
            'Estalvi Autoconsum (€)': estalvi_auto_cups,
            'Estalvi Compensació (€)': estalvi_comp_cups,
            'Estalvi Anual (€)': cost_nosolar_cups - cost_solar_cups,
            'Mensual': mensual_stats,
        })

        total_cost_nosolar += cost_nosolar_cups
        total_cost_solar += cost_solar_cups

    return results_by_cups, total_cost_nosolar - total_cost_solar


def render_cle_variable_optimizer():
    """
    Renderitza la pestanya de coeficients variables dins del CLE Pavelló.
    Requereix que st.session_state tingui les claus:
      - cle_df_consum, cle_gen_pav, cle_gen_sn, cle_prices_arr,
        cle_p_exc_val, cle_min_kwp_val
    (guardades per render_cle_optimizer() en clicar el botó principal)
    """
    st.markdown("### 🔀 Coeficients Variables Horaris")
    st.markdown("""
    Distribueix la quota del Pavelló **proporcionalment al consum real** de cada
    equipament en cada franja horària. Durant les nits d'estiu (Jun–Ago, 21h–07h)
    augmenta el pes dels CUPS d'enllumeament per aprofitar la generació nocturna
    residual i evitar excedents no compensables.

    > La Sala Nova manté el seu coeficient fix en tot moment.
    """)

    # Comprovar que hi ha dades disponibles (del botó principal)
    if not st.session_state.get('cle_df_consum') is not None and \
       st.session_state.get('cle_df_consum') is None:
        st.info("ℹ️ Primer executeu l'optimització des de la pestanya **📌 Coeficients Fixes** per carregar les dades.")
        return

    df_consum = st.session_state.get('cle_df_consum')
    gen_pav = st.session_state.get('cle_gen_pav')
    gen_sn = st.session_state.get('cle_gen_sn')
    prices = st.session_state.get('cle_prices_arr')
    p_exc = st.session_state.get('cle_p_exc_val', 0.07)
    min_kwp = st.session_state.get('cle_min_kwp_val', 0.5)

    if df_consum is None or gen_pav is None:
        st.info("ℹ️ Primer executeu l'optimització des de la pestanya **📌 Coeficients Fixes** per carregar les dades.")
        return

    # --- Controls ---
    boost_pct = st.slider(
        "🌙 Boost nocturn estiu (Jun–Ago, 21h–07h) cap a enllumeaments:",
        min_value=0, max_value=50, value=20, step=5,
        format="%d%%",
        help=(
            "Percentatge de la quota dels edificis que es redistribuirà "
            "als CUPS d'enllumeament en nits d'estiu (Jun-Ago). "
            "0% = repartiment purament proporcional al consum."
        ),
        key="var_boost_slider"
    )
    boost = boost_pct / 100.0

    if st.button("⚡ Calcular Coeficients Variables", key="btn_var_coef", type="primary"):
        with st.spinner("Calculant matriu de coeficients (12 mesos × 24 hores)..."):
            coef_matrix = compute_variable_coef_matrix(
                df_consum,
                summer_night_boost=boost,
                min_kwp_threshold=min_kwp,
            )
        with st.spinner("Simulant facturació hora per hora..."):
            results_var, total_savings_var = evaluate_variable_coefs(
                coef_matrix,
                list(df_consum.columns),
                df_consum, gen_pav, gen_sn, prices, p_exc,
            )
        st.session_state['cle_var_coef_matrix'] = coef_matrix
        st.session_state['cle_var_results'] = results_var
        st.session_state['cle_var_savings'] = total_savings_var
        st.success(
            f"✅ Coeficients calculats! Estalvi estimat: **{total_savings_var:,.0f} €**"
        )

    # --- Resultats ---
    if not st.session_state.get('cle_var_results'):
        return

    results_var: list = st.session_state['cle_var_results']
    coef_matrix: dict = st.session_state['cle_var_coef_matrix']
    total_savings_var: float = st.session_state['cle_var_savings']

    # KPIs globals
    tot_con = sum(r['Consum Anual (kWh)'] for r in results_var)
    tot_auto = sum(r['Autoconsum Total (kWh)'] for r in results_var)
    tot_exc_comp = sum(r['Excedents Compensats (kWh)'] for r in results_var)
    tot_exc_lost = sum(r['Excedents Llençats a la xarxa (kWh)'] for r in results_var)
    tot_gen = tot_auto + tot_exc_comp + tot_exc_lost
    pct_cob = (tot_auto / tot_con * 100) if tot_con > 0 else 0.0
    pct_apro = ((tot_auto + tot_exc_comp) / tot_gen * 100) if tot_gen > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💶 Estalvi Total", f"{total_savings_var:,.0f} €".replace(',', '.'))
    k2.metric("⚡ Autoconsum", f"{tot_auto:,.0f} kWh".replace(',', '.'))
    k3.metric("🛡️ Cobertura", f"{pct_cob:.1f} %")
    k4.metric("♻️ Aprofitament Planta", f"{pct_apro:.1f} %")

    # --- Comparativa vs. coeficients fixes (si disponible) ---
    if st.session_state.get('cle_results'):
        fixed_results: list = st.session_state['cle_results']
        tot_auto_fix = sum(r['Autoconsum Total (kWh)'] for r in fixed_results)
        tot_sav_fix = sum(r['Estalvi Anual (€)'] for r in fixed_results)
        d_auto_total = tot_auto - tot_auto_fix
        d_sav_total = total_savings_var - tot_sav_fix

        st.markdown("#### 📊 Comparativa: Variables vs. Fixes (SLSQP)")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric(
            "Δ Autoconsum total",
            f"{d_auto_total:+,.0f} kWh".replace(',', '.'),
            delta_color="normal" if d_auto_total >= 0 else "inverse"
        )
        dc2.metric(
            "Δ Estalvi total",
            f"{d_sav_total:+,.0f} €".replace(',', '.'),
            delta_color="normal" if d_sav_total >= 0 else "inverse"
        )
        dc3.metric("Estalvi Fixes (SLSQP)", f"{tot_sav_fix:,.0f} €".replace(',', '.'))

        fixed_by_cups = {r['CUPS']: r for r in fixed_results}
        comp_rows = []
        for r_v in results_var:
            cups = r_v['CUPS']
            r_f = fixed_by_cups.get(cups, {})
            d_auto = r_v['Autoconsum Total (kWh)'] - r_f.get('Autoconsum Total (kWh)', 0)
            d_sav = r_v['Estalvi Anual (€)'] - r_f.get('Estalvi Anual (€)', 0)
            comp_rows.append({
                'Nom': r_v['Nom'],
                'Coef Fix': f"{r_f.get('Coeficient Pavelló', 0):.6f}",
                'Coef Var Mig': f"{r_v['Coeficient Pavelló (Mig)']:.6f}",
                'Δ Autoconsum (kWh)': f"{d_auto:+,.0f}".replace(',', '.'),
                'Δ Estalvi (€)': f"{d_sav:+,.2f}".replace(',', '.'),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        st.markdown("---")

    # --- Heatmaps ---
    st.markdown("#### 🔥 Heatmaps de Coeficients")
    cups_ids_list = list(coef_matrix.keys())
    cups_names_disp = [CUPS_MAPPING.get(c, c) for c in cups_ids_list]

    col_h1, col_h2 = st.columns([1, 2])
    with col_h1:
        selected_month = st.selectbox(
            "Mes pel heatmap hora×CUPS:",
            range(1, 13),
            format_func=lambda x: MESOS_CAT[x - 1],
            key="var_heatmap_month"
        )

    # Heatmap hora×CUPS
    hm_data = np.array([coef_matrix[c][selected_month - 1, :] for c in cups_ids_list])
    fig_hm1 = go.Figure(go.Heatmap(
        z=hm_data,
        x=[f"{h:02d}h" for h in range(24)],
        y=cups_names_disp,
        colorscale='Viridis',
        colorbar=dict(title="Coef."),
        hovertemplate="CUPS: %{y}<br>Hora: %{x}<br>Coef: %{z:.6f}<extra></extra>",
    ))
    fig_hm1.update_layout(
        title=f"Coeficients per Hora — {MESOS_CAT[selected_month - 1]}",
        xaxis_title="Hora del dia",
        yaxis_title="CUPS",
        height=max(300, len(cups_ids_list) * 38 + 120),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hm1, use_container_width=True)

    # Heatmap mes×CUPS (coeficient mig per mes)
    hm_monthly = np.array([coef_matrix[c].mean(axis=1) for c in cups_ids_list])
    fig_hm2 = go.Figure(go.Heatmap(
        z=hm_monthly,
        x=MESOS_CAT,
        y=cups_names_disp,
        colorscale='Viridis',
        colorbar=dict(title="Coef mig"),
        hovertemplate="CUPS: %{y}<br>Mes: %{x}<br>Coef Mig: %{z:.6f}<extra></extra>",
    ))
    fig_hm2.update_layout(
        title="Coeficient Mig per Mes × CUPS",
        xaxis_title="Mes",
        yaxis_title="CUPS",
        height=max(300, len(cups_ids_list) * 38 + 120),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_hm2, use_container_width=True)

    # Gràfic barres mensual (autoconsum SN + PAV per mes)
    st.markdown("#### 📊 Energia Mensual (Global)")
    monthly_aggs = []
    for m in range(1, 13):
        m_con = sum(
            next(x for x in r['Mensual'] if x['Mes'] == m)['Consum']
            for r in results_var
        )
        m_auto_sn = sum(
            next(x for x in r['Mensual'] if x['Mes'] == m)['Autoconsum SN']
            for r in results_var
        )
        m_auto_pav = sum(
            next(x for x in r['Mensual'] if x['Mes'] == m)['Autoconsum PAV']
            for r in results_var
        )
        m_import = sum(
            next(x for x in r['Mensual'] if x['Mes'] == m)['Import Net']
            for r in results_var
        )
        m_estalvi = sum(
            next(x for x in r['Mensual'] if x['Mes'] == m)['Estalvi € (Brut)']
            for r in results_var
        )
        monthly_aggs.append({
            'Mes': MESOS_CAT[m - 1],
            'Import (kWh)': m_import,
            'Autoconsum SN (kWh)': m_auto_sn,
            'Autoconsum PAV (kWh)': m_auto_pav,
            'Estalvi € (Brut)': m_estalvi,
        })

    df_m = pd.DataFrame(monthly_aggs)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_m['Mes'], y=df_m['Import (kWh)'],
        name='Importació (Pagada)', marker_color='#a5d6a7'
    ))
    fig_bar.add_trace(go.Bar(
        x=df_m['Mes'], y=df_m['Autoconsum SN (kWh)'],
        name='Autoconsum Sala Nova', marker_color='#fbc02d'
    ))
    fig_bar.add_trace(go.Bar(
        x=df_m['Mes'], y=df_m['Autoconsum PAV (kWh)'],
        name='Autoconsum Pavelló (Variable)', marker_color='#1565c0'
    ))
    fig_bar.update_layout(
        barmode='stack',
        title="Estructura de l'Energia Mensual (Coeficients Variables)",
        xaxis_title='Mes', yaxis_title='kWh',
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_line = px.line(
        df_m, x='Mes', y='Estalvi € (Brut)', markers=True,
        title='Estalvi Efectiu Mensual (amb Taxes i Límits RD 244)',
    )
    fig_line.update_traces(line_color='#d32f2f', marker=dict(size=8))
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Taula de resultats per CUPS ---
    st.markdown("#### 📋 Resultats per Equipament")
    df_res = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'Mensual'}
        for r in results_var
    ])
    # Fila de totals
    import numpy as _np
    num_cols = df_res.select_dtypes(include=[_np.number]).columns
    tot_row = {c: df_res[c].sum() for c in num_cols}
    tot_row['CUPS'] = 'TOTAL AGREGAT'
    tot_row['Nom'] = ''
    tot_c = tot_row.get('Consum Anual (kWh)', 0)
    tot_a = tot_row.get('Autoconsum Total (kWh)', 0)
    tot_row['Cobertura (%)'] = (tot_a / tot_c * 100) if tot_c > 0 else 0
    df_res = pd.concat([df_res, pd.DataFrame([tot_row])], ignore_index=True)
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # --- CSV oficial (matriu de coeficients) ---
    st.markdown("#### 📥 Exportació CSV Official")
    csv_rows = []
    for cups_id in cups_ids_list:
        for m_idx in range(12):
            for h in range(24):
                csv_rows.append({
                    'CUPS': cups_id,
                    'Nom': CUPS_MAPPING.get(cups_id, ''),
                    'Mes': m_idx + 1,
                    'Hora': h,
                    'Coeficient_Pavello': f"{coef_matrix[cups_id][m_idx, h]:.6f}",
                })
    df_csv = pd.DataFrame(csv_rows)
    csv_bytes = df_csv.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
    st.download_button(
        "📥 Descarregar Matriu de Coeficients Variables (CSV oficial RD 244/2019)",
        data=csv_bytes,
        file_name="cle_pavello_coeficients_variables.csv",
        mime='text/csv',
    )
