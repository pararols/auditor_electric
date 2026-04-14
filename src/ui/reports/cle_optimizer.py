import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from pathlib import Path
from src.core.config import COMMUNITY_QUOTAS, COMMUNITY_PARTICIPANTS, CUPS_MAPPING
from src.core.database import load_from_supabase_db, get_last_complete_day_all_cups

# Paràmetres globals Subapp
TARGET_KWP = 78.198
PAVELLO_KWP = 127.2
TARGET_RATIO = TARGET_KWP / PAVELLO_KWP

def load_solar_curves():
    """Carrega les corbes de generació anual del Pavelló (127.2 kWp) i Sala Nova (17.1 kWp).
    Retorna dos Series amb índex Datetime (any 2025 fictici per a mapping).
    """
    base_path = Path(__file__).parents[3]
    
    # Pavello
    df_pav = pd.read_csv(base_path / "modelpavello.csv", sep=";", decimal=",")
    df_pav = df_pav.dropna(subset=['Produccio FV kWh'])
    
    # Sala nova
    df_sala = pd.read_csv(base_path / "modelsalanova.csv", sep=";", decimal=",")
    df_sala = df_sala.dropna(subset=['Produccio FV kWh'])
    
    # Crear índex temporal 2025 (no traspàs) per a 8760 hores
    idx_2025 = pd.date_range("2025-01-01 00:00:00", "2025-12-31 23:00:00", freq="h")
    
    gen_pavello = pd.Series(df_pav['Produccio FV kWh'].values[:8760], index=idx_2025)
    gen_salanova = pd.Series(df_sala['Produccio FV kWh'].values[:8760], index=idx_2025)
    
    return gen_pavello, gen_salanova

@st.cache_data(ttl=3600)
def fetch_and_prep_consumption(start_date, end_date):
    """Carrega el consum horari per a un període de 1 any per als CUPS municipals."""
    # start_date, end_date són objectes date o strings 'YYYY-MM-DD'
    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    
    # Carreguem dades de Supabase pel rang especificat
    df_raw = load_from_supabase_db(start_date=start_str, end_date=end_str)
    
    if df_raw is None or df_raw.empty:
        return None
    
    # Reconstruir el consum total (Xarxa + Autoconsum existent de Sala Nova)
    cups_data = {}
    municipal_cups_ids = [k for k, v in CUPS_MAPPING.items() if not str(v).startswith("Part. Privat")]
    for cups_id in municipal_cups_ids:
        cups_name = CUPS_MAPPING.get(cups_id)
        
        # Check if the friendly name is in the DB, fallback to ID just in case
        if cups_name in df_raw.columns.get_level_values(0):
            target_col = cups_name
        elif cups_id in df_raw.columns.get_level_values(0):
            target_col = cups_id
        else:
            continue
            
        cols = df_raw[target_col].columns
        ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
        auto_col = [c for c in cols if 'AUTOCONS' in c]
        
        val_total = pd.Series(0.0, index=df_raw.index)
        if ae_col:
            val_total = val_total.add(df_raw[target_col][ae_col[0]], fill_value=0)
        if auto_col:
            val_total = val_total.add(df_raw[target_col][auto_col[0]], fill_value=0)
            
        cups_data[cups_id] = val_total
        
    df_final = pd.DataFrame(cups_data)
    
    # Generar índex complet per a l'any seleccionat (8760-8784 hores depenent de traspàs)
    # Per simplificar a l'optimitzador, forçarem 8760 punts basant-nos en el rang
    full_index = pd.date_range(start=f"{start_str} 00:00:00", end=f"{end_str} 23:00:00", freq='h')
    
    # Si és any de traspàs o hi ha desviaments horaris, re-calculem per tenir un any estàndard per a la comparativa solar
    # (L'algorisme està dissenyat per a vectors de 8760)
    df_final = df_final.groupby(df_final.index).mean()
    df_final = df_final.reindex(full_index, fill_value=0)
    
    # Ens quedem amb exactament 8760 hores per al motor
    if len(df_final) > 8760:
        df_final = df_final.iloc[:8760]
    elif len(df_final) < 8760:
        # Pad with zeros if data is incomplete
        pad_size = 8760 - len(df_final)
        pad_df = pd.DataFrame(0.0, index=range(pad_size), columns=df_final.columns)
        df_final = pd.concat([df_final, pad_df]).fillna(0)
    
    return df_final

def calculate_tariffs(full_index, p1=0.22, p2=0.147, p3=0.11):
    """Construeix el vector de tarifes P1/P2/P3 segons calendari peninsular."""
    # Lògica senzilla per blocs (3.0TD té 6 periodes, 2.0TD en té 3). 
    # Suposarem la 2.0TD o una simplificació: 
    # P3: 00h-08h i caps de setmana tot el dia.
    # P2: 08h-10h, 14h-18h, 22h-00h de dilluns a divendres.
    # P1: 10h-14h i 18h-22h de dilluns a divendres.
    
    prices = np.zeros(len(full_index))
    
    for i, dt in enumerate(full_index):
        if dt.weekday() >= 5: # Cap de setmana
            prices[i] = p3
        else: # Dies feiners
            h = dt.hour
            if h < 8:
                prices[i] = p3
            elif (8 <= h < 10) or (14 <= h < 18) or (22 <= h < 24):
                prices[i] = p2
            else:
                prices[i] = p1
                
    return prices

def evaluate_coefficients(coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price):
    """
    Simula la facturació anual neta dels CUPS donats uns coeficients del pavelló.
    Aplica el topall mensual legal i els impostos (5.11% electricitat + 21% IVA).
    L'estalvi total = cost_sense_solar - cost_amb_solar.
    Retorna l'estalvi (-estalvi per a minimitzar a Scipy), o els resultats detallats.
    """
    total_cost_nosolar = 0
    total_cost_solar = 0
    
    results_by_cups = []
    
    months = df_consum.index.month
    
    for i, cups in enumerate(cups_names):
        consum = df_consum[cups].values
        
        # Quota fixada a configuració per Sala Nova (si hi està present)
        sn_info = COMMUNITY_QUOTAS.get(cups)
        if sn_info:
            coef_sn = sn_info['coef']
        else:
            coef_sn = 0.0
            
        coef_pav = coefs[i]
        
        # Generació assignada
        gen_sn = gen_salanova * coef_sn
        gen_pav = gen_pavello * coef_pav
        gen_total = gen_sn + gen_pav
        
        # Autoconsum i Excedents
        autoconsum = np.minimum(consum, gen_total)
        net_import = consum - autoconsum
        excedents = gen_total - autoconsum
        
        cost_nosolar_cups = 0
        cost_solar_cups = 0
        
        estalvi_auto_cups = 0
        estalvi_comp_cups = 0
        
        excedents_compensats_qty = 0
        excedents_abocats_qty = 0
        
        mensual_stats = []
        
        # Càlcul mes a mes per aplicar el límit RD 244/2019
        for m in range(1, 13):
            mask = (months == m)
            
            # Sense solar
            c_nos = np.sum(consum[mask] * prices[mask])
            c_nos_taxes = (c_nos * 1.0511) * 1.21
            cost_nosolar_cups += c_nos_taxes
            
            # Amb solar
            c_sol_import = np.sum(net_import[mask] * prices[mask])
            v_excedents = np.sum(excedents[mask] * excedent_price)
            
            # Topall legal compensació: "El terme d'energia no pot ser inferior a 0 abans d'impostos"
            terme_energia_net = max(0, c_sol_import - v_excedents)
            
            # Impostos aplicats SOBRE EL TERME NET
            c_sol_taxes = (terme_energia_net * 1.0511) * 1.21
            cost_solar_cups += c_sol_taxes
            
            # Desglossament d'estalvis
            estalvi_auto_month = (c_nos - c_sol_import) * 1.0511 * 1.21
            estalvi_comp_month = (c_sol_import - terme_energia_net) * 1.0511 * 1.21
            
            estalvi_auto_cups += estalvi_auto_month
            estalvi_comp_cups += estalvi_comp_month
            
            # Tracking físic
            if v_excedents <= c_sol_import:
                exc_comp = np.sum(excedents[mask])
                exc_lost = 0
            else:
                # Hem topat. La part d'energia que hem cobrat virtualment és c_sol_import / excedent_price
                exc_comp = c_sol_import / excedent_price if excedent_price > 0 else 0
                exc_lost = np.sum(excedents[mask]) - exc_comp
            
            excedents_compensats_qty += exc_comp
            excedents_abocats_qty += exc_lost
            
            # Repartiment proporcional d'Autoconsum entre plantes
            prop_sn = np.zeros_like(gen_total[mask])
            prop_pav = np.zeros_like(gen_total[mask])
            valid_gen = gen_total[mask] > 0
            prop_sn[valid_gen] = gen_sn[mask][valid_gen] / gen_total[mask][valid_gen]
            prop_pav[valid_gen] = gen_pav[mask][valid_gen] / gen_total[mask][valid_gen]
            
            auto_sn_mask = autoconsum[mask] * prop_sn
            auto_pav_mask = autoconsum[mask] * prop_pav
            
            mensual_stats.append({
                'Mes': m,
                'Consum': np.sum(consum[mask]),
                'Generació Assignada': np.sum(gen_total[mask]),
                'Generació Assignada SN': np.sum(gen_sn[mask]),
                'Generació Assignada PAV': np.sum(gen_pav[mask]),
                'Autoconsum SN': np.sum(auto_sn_mask),
                'Autoconsum PAV': np.sum(auto_pav_mask),
                'Autoconsum': np.sum(autoconsum[mask]),
                'Import Net': np.sum(net_import[mask]),
                'Estalvi € (Brut)': c_nos_taxes - c_sol_taxes
            })
            
        total_auto_sn = sum([x['Autoconsum SN'] for x in mensual_stats])
        total_auto_pav = sum([x['Autoconsum PAV'] for x in mensual_stats])
        
        results_by_cups.append({
            'CUPS': cups,
            'Nom': CUPS_MAPPING.get(cups, "Desconegut"),
            'Coeficient Sala Nova': coef_sn,
            'Potència Sala Nova (kWp)': coef_sn * 17.1,  # Based on 17.1 total
            'Coeficient Pavelló': coef_pav,
            'Potència Pavelló (kWp)': coef_pav * PAVELLO_KWP,
            'Consum Anual (kWh)': np.sum(consum),
            'Producció FV (kWh)': np.sum(gen_total),
            'Autoconsum SN (kWh)': total_auto_sn,
            'Autoconsum PAV (kWh)': total_auto_pav,
            'Autoconsum Total (kWh)': np.sum(autoconsum),
            'Cobertura (%)': (np.sum(autoconsum) / np.sum(consum)) * 100 if np.sum(consum)>0 else 0,
            'Excedents Compensats (kWh)': excedents_compensats_qty,
            'Excedents Llençats a la xarxa (kWh)': excedents_abocats_qty,
            'Estalvi Autoconsum (€)': estalvi_auto_cups,
            'Estalvi Compensació (€)': estalvi_comp_cups,
            'Estalvi Anual (€)': cost_nosolar_cups - cost_solar_cups,
            'Mensual': mensual_stats
        })
        
        total_cost_nosolar += cost_nosolar_cups
        total_cost_solar += cost_solar_cups
        
    total_savings = total_cost_nosolar - total_cost_solar
    return -total_savings, results_by_cups

def objective_function(coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price):
    savings_neg, _ = evaluate_coefficients(coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price)
    return savings_neg

def run_optimization(df_consum, prices, excedent_price, min_kwp_threshold=0.0):
    cups_names = list(df_consum.columns)
    N = len(cups_names)
    
    # Carregar corbes model (8760h)
    gen_pav_model, gen_sn_model = load_solar_curves()
    
    # Alineació Intel·ligent: Mapejar cada hora del consum a la seva hora equivalent del model solar
    # (Ignorem l'any i ens fixem en Mes-Dia-Hora)
    def align_solar(model_series, target_index):
        # Creem una clau 'month-day-hour' per al model
        m_keys = model_series.index.strftime('%m-%d-%H')
        m_map = dict(zip(m_keys, model_series.values))
        
        # Mapegem l'índex de consum
        t_keys = target_index.strftime('%m-%d-%H')
        return np.array([m_map.get(k, 0.0) for k in t_keys])

    gen_pavello = align_solar(gen_pav_model, df_consum.index)
    gen_salanova = align_solar(gen_sn_model, df_consum.index)
    
    # Valors inicials igualitaris
    init_guess = np.ones(N) * (TARGET_RATIO / N)
    
    # Límits actius variables (podem forçar 0 si pertoca)
    active_bounds = [(0, 1) for _ in range(N)]
    
    # Restricció: suma de coeficients = TARGET_RATIO
    constraints = {'type': 'eq', 'fun': lambda c: np.sum(c) - TARGET_RATIO}
    
    # --- Interfície de Temps Real ---
    st.markdown("#### 🔄 Progrés de l'Optimització (Temps Real)")
    col_it, col_sav, col_aut, col_exc_comp, col_exc_lost = st.columns(5)
    pl_it = col_it.empty()
    pl_sav = col_sav.empty()
    pl_aut = col_aut.empty()
    pl_exc_comp = col_exc_comp.empty()
    pl_exc_lost = col_exc_lost.empty()
    
    iteration_count = [0]
    
    def optimizer_callback(xk):
        iteration_count[0] += 1
        sav_neg, details = evaluate_coefficients(xk, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price)
        
        t_sav = -sav_neg
        t_aut = sum(d['Autoconsum Total (kWh)'] for d in details)
        t_exc_comp = sum(d['Excedents Compensats (kWh)'] for d in details)
        t_exc_lost = sum(d['Excedents Llençats a la xarxa (kWh)'] for d in details)
        
        pl_it.metric("Iteració", f"#{iteration_count[0]}")
        pl_sav.metric("Estalvi Anual", f"{t_sav:,.0f} €".replace(',', '.'))
        pl_aut.metric("Autoconsum", f"{t_aut:,.0f} kWh".replace(',', '.'))
        pl_exc_comp.metric("Excedent Compensat", f"{t_exc_comp:,.0f} kWh".replace(',', '.'))
        pl_exc_lost.metric("Excedent No Compensat", f"{t_exc_lost:,.0f} kWh".replace(',', '.'))

    with st.spinner("L'algoritme d'optimització iteratiu està determinant el repartiment màxim... (això pot trigar uns segons)"):
        opt_res = minimize(
            objective_function, 
            init_guess,
            args=(cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price),
            method='SLSQP',
            bounds=active_bounds,
            constraints=constraints,
            callback=optimizer_callback,
            options={'disp': False, 'ftol': 1e-4}
        )
        
        # Filtre actiu per micro-assignacions interactiu
        if min_kwp_threshold > 0:
            while True:
                kwp_assigned = opt_res.x * PAVELLO_KWP
                # Buscar aquells que són positius però insuficients
                to_exclude = (kwp_assigned > 1e-6) & (kwp_assigned < min_kwp_threshold)
                if not np.any(to_exclude):
                    break
                
                # Forçar aquests coeficients a 0
                for i in range(N):
                    if to_exclude[i]:
                        active_bounds[i] = (0, 0)
                        
                # Nou punt inicial sa per SLSQP
                active_count = sum(1 for b in active_bounds if b[1] == 1)
                new_guess = np.zeros(N)
                if active_count > 0:
                    for i in range(N):
                        if active_bounds[i][1] == 1:
                            new_guess[i] = TARGET_RATIO / active_count
                            
                opt_res = minimize(
                    objective_function, 
                    new_guess,
                    args=(cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price),
                    method='SLSQP',
                    bounds=active_bounds,
                    constraints=constraints,
                    callback=optimizer_callback,
                    options={'disp': False, 'ftol': 1e-4}
                )
        
    # --- Ajust d'Arrodoniment a 6 Decimals (Legalitat RD 244/2019) ---
    raw_coefs = opt_res.x
    target_int = int(round(TARGET_RATIO * 1e6))
    
    coefs_int = np.int64(np.floor(raw_coefs * 1e6))
    remainders = (raw_coefs * 1e6) - coefs_int
    
    diff = target_int - np.sum(coefs_int)
    if diff > 0:
        # Repartir el diff restant als que tenen el remainder més gran
        indices = np.argsort(remainders)[-diff:]
        for idx in indices:
            coefs_int[idx] += 1
            
    final_coefs = coefs_int / 1e6
        
    # Extraure resultats finals
    _, detailed_results = evaluate_coefficients(
        final_coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price
    )
    return detailed_results

    
def render_cle_optimizer():
    # Estils d'impressió CSS (Amagar barra lateral i botons)
    st.markdown("""
        <style>
        @media print {
            section[data-testid="stSidebar"], 
            .stHeader, 
            header,
            footer,
            .stButton,
            div[data-testid="stDownloadButton"] {
                display: none !important;
            }
            .main .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("⚙️ Optimitzador CLE: Assignació Repartiment Pavelló (RD 244/2019)")
    st.markdown("""
        Aquesta subaplicació determina automàticament els **coeficients fixes ideals anuals** per a la nova instal·lació **FV Pavelló (127.2 kWp)**, amb l'objectiu de distribuir exclusivament la quota reservada per a l'Ajuntament (78.198 kWp).
        
        **Criteris aplicats segons normativa ESP:**
        * L'energia provinent de la instal·lació Sala Nova es té en compte prèviament per calcular consums romanents.
        * Simulació amb freqüència **horària** utilitzant històrìcs de 2025 i extrapolació de les dues corbes FV.
        * El límit de compensació d'excedents mensual s'aplica estrictament abans d'impostos.
        * Totes les xifres econòmiques estan finalment gravades amb Impost Elèctric (5.11%) i IVA (21%).
    """)
    
    # --- GESTIÓ D'ESTAT (SESSION STATE) ---
    if 'cle_results' not in st.session_state:
        st.session_state.cle_results = None
    if 'cle_end_date' not in st.session_state:
        st.session_state.cle_end_date = None
    
    st.markdown("#### Paràmetres d'Anàlisi i Opcions Legals")
    
    # 1. Trobar data límit per defecte
    municipal_cups_ids = [k for k, v in CUPS_MAPPING.items() if not str(v).startswith("Part. Privat")]
    default_end_date = get_last_complete_day_all_cups(municipal_cups_ids)
    
    # 2. UI Selector de Període
    col0, col1, col2, col3, col4 = st.columns([2, 1, 1, 1, 1.5])
    with col0:
        current_end_date = st.date_input("Analitzar període de 1 any fins a:", value=default_end_date or pd.Timestamp.now().date())
        start_date = current_end_date - pd.Timedelta(days=364)
        st.info(f"Rang d'anàlisi: {start_date.strftime('%d/%m/%Y')} - {current_end_date.strftime('%d/%m/%Y')}")

    # Nova sub-opció ben visible a dalt (Poda de <0.5 kWp)
    filter_micro = col4.checkbox("Excloure < 0.5 kWp", value=True, help="Simplifica la gestió administrativa municipal")
    min_kwp_val = 0.5 if filter_micro else 0.0
    
    st.write("") # Separador
    
    # UI Constants
    p1 = col1.number_input("Preu P1 (€)", value=0.22, format="%.3f")
    p2 = col2.number_input("Preu P2 (€)", value=0.147, format="%.3f")
    p3 = col3.number_input("Preu P3 (€)", value=0.11, format="%.3f")
    # p_exc = col4.number_input("Comp. (€)", value=0.07, format="%.3f") # Removed col4 to refactor
    
    col_a, col_b = st.columns([1, 4])
    p_exc = col_a.number_input("Compensació Excedents (€)", value=0.07, format="%.3f")

    st.write("") # Separador
    
    col_run, col_print = st.columns([1, 4])
    run_btn = col_run.button("Executar Motor d'Optimització", type="primary")
    
    # Botó d'impressió amb JS
    if col_print.button("🖨️ Imprimir resultats"):
        st.components.v1.html("""
            <script>
                window.print();
            </script>
        """, height=0)

    if run_btn:
        df_consum = fetch_and_prep_consumption(start_date, current_end_date)
        if df_consum is None or df_consum.empty:
            st.error(f"No s'han trobat dades de consum suficients per al període seleccionat.")
            return
            
        prices = calculate_tariffs(df_consum.index, p1, p2, p3)
        
        # Corre l'optimització amb el paràmetre de tall de kWp
        st.session_state.cle_results = run_optimization(df_consum, prices, p_exc, min_kwp_threshold=min_kwp_val)
        st.session_state.cle_end_date = current_end_date
        st.success("Optimització finalitzada amb èxit!")

    # --- RENDERITZAT DE RESULTATS (DES DE L'ESTAT) ---
    if st.session_state.cle_results is not None:
        detailed_results = st.session_state.cle_results
        saved_end_date = st.session_state.cle_end_date
        
        # --- 1. RESUM EXECUTIU ---
        st.markdown("### 📊 Resum Executiu Global")
        
        # Extracció de Totals Absoluts
        tot_consum = sum([r['Consum Anual (kWh)'] for r in detailed_results])
        tot_auto = sum([r['Autoconsum Total (kWh)'] for r in detailed_results])
        tot_exc_comp = sum([r['Excedents Compensats (kWh)'] for r in detailed_results])
        tot_exc_lost = sum([r['Excedents Llençats a la xarxa (kWh)'] for r in detailed_results])
        tot_gen = tot_auto + tot_exc_comp + tot_exc_lost
        
        tot_est_auto = sum([r['Estalvi Autoconsum (€)'] for r in detailed_results])
        tot_est_comp = sum([r['Estalvi Compensació (€)'] for r in detailed_results])
        tot_est_global = sum([r['Estalvi Anual (€)'] for r in detailed_results])
        
        # Càlcul de Percentatges (%)
        pct_cobertura = (tot_auto / tot_consum * 100) if tot_consum > 0 else 0
        pct_aprofitament = ((tot_auto + tot_exc_comp) / tot_gen * 100) if tot_gen > 0 else 0
        
        # KPI Cards
        st.write("")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("💶 Estalvi Econòmic Total", f"{tot_est_global:,.0f} €".replace(',', '.'))
        with col_k2:
            st.metric("⚡ Energia Generada (Quota)", f"{tot_gen:,.0f} kWh".replace(',', '.'))
        with col_k3:
            st.metric("🛡️ Cobertura Autoconsum", f"{pct_cobertura:.1f} %")
        with col_k4:
            st.metric("♻️ Aprofitament de Planta", f"{pct_aprofitament:.1f} %", help="Inclou Autoconsum + Excedents Compensats")
            
        st.write("")
        
        # Gràfics de Donut Professionals
        col_g1, col_g2, col_g3 = st.columns(3)
        
        # Gràfic 1: Origen de l'Estalvi
        fig_donut1 = px.pie(
            names=["Estalvi per Autoconsum", "Descompte per Excedents"],
            values=[tot_est_auto, tot_est_comp],
            hole=0.6,
            color_discrete_sequence=["#2ecc71", "#f1c40f"]
        )
        fig_donut1.update_scenes(aspectratio=dict(x=1, y=1, z=1))
        fig_donut1.update_layout(title_text="Desglossament Econòmic (€)", title_x=0.5, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        fig_donut1.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}<br>%{value:,.0f} €<extra></extra>")
        col_g1.plotly_chart(fig_donut1, use_container_width=True)
        
        # Gràfic 2: Destí de la Producció
        fig_donut2 = px.pie(
            names=["Autoconsum directe", "Excedent Compensat (Venut)", "Excedent Abocat (Perdut)"],
            values=[tot_auto, tot_exc_comp, tot_exc_lost],
            hole=0.6,
            color_discrete_sequence=["#27ae60", "#f39c12", "#e74c3c"]
        )
        fig_donut2.update_layout(title_text="Destí Energia Generada (kWh)", title_x=0.5, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        fig_donut2.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}<br>%{value:,.0f} kWh<extra></extra>")
        col_g2.plotly_chart(fig_donut2, use_container_width=True)
        
        # Gràfic 3: Origen del Consum
        compra_xarxa = max(0, tot_consum - tot_auto)
        fig_donut3 = px.pie(
            names=["Autoconsum Solar", "Compra a la Xarxa"],
            values=[tot_auto, compra_xarxa],
            hole=0.6,
            color_discrete_sequence=["#2ecc71", "#34495e"]
        )
        fig_donut3.update_layout(title_text="Procedència del Consum (kWh)", title_x=0.5, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        fig_donut3.update_traces(textposition='inside', textinfo='percent+label', hovertemplate="%{label}<br>%{value:,.0f} kWh<extra></extra>")
        col_g3.plotly_chart(fig_donut3, use_container_width=True)

        st.markdown("---")
        
        # --- 2. TAULA DETALLADA DE PUNTS DE SUBMINISTRAMENT ---
        st.markdown("### 📋 Resultats Detallats per Equipament")
        
        df_res = pd.DataFrame(detailed_results)
        df_res = df_res.drop(columns=['Mensual'])
        
        # Add Total Row
        totals = df_res.sum(numeric_only=True)
        totals['CUPS'] = 'TOTAL AGREGAT'
        totals['Nom'] = ''
        totals['Cobertura (%)'] = (totals['Autoconsum Total (kWh)'] / totals['Consum Anual (kWh)']) * 100 if totals['Consum Anual (kWh)'] else 0
        df_res = pd.concat([df_res, pd.DataFrame([totals])], ignore_index=True)
        
        # CSV Export Preparation (Before string formatting)
        csv_data = df_res.to_csv(index=False, sep=';', decimal=',')
        st.download_button(
            label="📥 Descarregar Resultats (CSV)",
            data=csv_data.encode('utf-8-sig'),
            file_name=f'resultats_cle_optimitzats_{saved_end_date.year}.csv',
            mime='text/csv'
        )
        
        # Format for Display
        df_display = df_res.copy()
        df_display['Coeficient Sala Nova'] = df_display['Coeficient Sala Nova'].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "")
        df_display['Potència Sala Nova (kWp)'] = df_display['Potència Sala Nova (kWp)'].apply(lambda x: f"{x:.2f} kWp" if pd.notnull(x) else "")
        df_display['Coeficient Pavelló'] = df_display['Coeficient Pavelló'].apply(lambda x: f"{x:.6f}")
        df_display['Potència Pavelló (kWp)'] = df_display['Potència Pavelló (kWp)'].apply(lambda x: f"{x:.2f} kWp" if pd.notnull(x) else "")
        
        df_display['Consum Anual (kWh)'] = df_display['Consum Anual (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_display['Producció FV'] = df_display['Producció FV (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        
        df_display['Estalvi Autoconsum (€)'] = df_display['Estalvi Autoconsum (€)'].apply(lambda x: f"{x:,.2f} €".replace(',','.'))
        df_display['Estalvi Compensació (€)'] = df_display['Estalvi Compensació (€)'].apply(lambda x: f"{x:,.2f} €".replace(',','.'))
        df_display['Estalvi Anual (€)'] = df_display['Estalvi Anual (€)'].apply(lambda x: f"{x:,.2f} €".replace(',','.'))
        
        df_display['Cobertura (%)'] = df_display['Cobertura (%)'].apply(lambda x: f"{x:.1f} %")
        df_display['Autoconsum (Sala Nova)'] = df_display['Autoconsum SN (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_display['Autoconsum (Pavelló)'] = df_display['Autoconsum PAV (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        
        df_display['Excedents Compensats'] = df_display['Excedents Compensats (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_display['Excedents Llençats'] = df_display['Excedents Llençats a la xarxa (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_display['Autoconsum Total'] = df_display['Autoconsum Total (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        
        # Defineix l'ordre final
        cols_order = [
            'CUPS', 'Nom', 
            'Coeficient Sala Nova', 'Potència Sala Nova (kWp)', 
            'Coeficient Pavelló', 'Potència Pavelló (kWp)',
            'Consum Anual (kWh)', 'Producció FV',
            'Autoconsum (Sala Nova)', 'Autoconsum (Pavelló)', 'Autoconsum Total', 'Cobertura (%)',
            'Excedents Compensats', 'Excedents Llençats',
            'Estalvi Autoconsum (€)', 'Estalvi Compensació (€)', 'Estalvi Anual (€)'
        ]
        df_display = df_display[cols_order]
        
        st.dataframe(df_display, use_container_width=True)
        
        # --- DASHBOARD MENSUAL ---
        st.markdown("### 📊 Dashboard Analític Mensual (Global Ajuntament)")
        
        # Agregar mensualitats de tots els CUPS
        monthly_aggs = []
        for m in range(1, 13):
            m_consum = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Consum'] for r in detailed_results])
            m_gen_sn = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Generació Assignada SN'] for r in detailed_results])
            m_gen_pav = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Generació Assignada PAV'] for r in detailed_results])
            m_auto_sn = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Autoconsum SN'] for r in detailed_results])
            m_auto_pav = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Autoconsum PAV'] for r in detailed_results])
            m_import = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Import Net'] for r in detailed_results])
            m_estalvi = sum([list(filter(lambda x: x['Mes'] == m, r['Mensual']))[0]['Estalvi € (Brut)'] for r in detailed_results])
            
            monthly_aggs.append({
                'Mes': m,
                'Consum Brut (kWh)': m_consum,
                'Import Net Xarxa (kWh)': m_import,
                'Generació SN (kWh)': m_gen_sn,
                'Generació PAV (kWh)': m_gen_pav,
                'Autoconsum SN (kWh)': m_auto_sn,
                'Autoconsum PAV (kWh)': m_auto_pav,
                'Estalvi Efectiu Mensual (€)': m_estalvi
            })
            
        df_months = pd.DataFrame(monthly_aggs)
        mesos_cat = ['Gen', 'Feb', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Oct', 'Nov', 'Des']
        df_months['Mes'] = df_months['Mes'].apply(lambda x: mesos_cat[x-1])
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df_months['Mes'], y=df_months['Import Net Xarxa (kWh)'], name='Importació (Pagada)', marker_color='#a5d6a7'))
        fig1.add_trace(go.Bar(x=df_months['Mes'], y=df_months['Autoconsum SN (kWh)'], name='Autoconsum (Sala Nova)', marker_color='#fbc02d'))
        fig1.add_trace(go.Bar(x=df_months['Mes'], y=df_months['Autoconsum PAV (kWh)'], name='Autoconsum (Pavelló)', marker_color='#2e7d32'))
        
        fig1.update_layout(barmode='stack', title='Estructura de l\'Energia Mensual per a l\'Administració',
                           xaxis_title='Mes', yaxis_title='kWh')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = go.Figure()
        
        exc_sn = df_months['Generació SN (kWh)'] - df_months['Autoconsum SN (kWh)']
        exc_pav = df_months['Generació PAV (kWh)'] - df_months['Autoconsum PAV (kWh)']
        
        fig3.add_trace(go.Bar(x=df_months['Mes'], y=df_months['Autoconsum SN (kWh)'] + df_months['Autoconsum PAV (kWh)'], name='Autoconsum Aprofitat', marker_color='#43a047'))
        fig3.add_trace(go.Bar(x=df_months['Mes'], y=exc_sn, name='Excedents No Aprofitats (Sala Nova)', marker_color='#fff59d'))
        fig3.add_trace(go.Bar(x=df_months['Mes'], y=exc_pav, name='Excedents No Aprofitats (Pavelló)', marker_color='#a5d6a7'))
        
        fig3.update_layout(barmode='stack', title='Destí de la Generació Solar: Aprofitament vs Excedents',
                           xaxis_title='Mes', yaxis_title='kWh')
        st.plotly_chart(fig3, use_container_width=True)
        
        fig2 = px.line(df_months, x='Mes', y='Estalvi Efectiu Mensual (€)', markers=True, 
                       title='Corba Econòmica: Estalvi Efectiu (Taxes i Límits RD 244 Inclòs)')
        fig2.update_traces(line_color='#d32f2f', marker=dict(size=8))
        st.plotly_chart(fig2, use_container_width=True)

