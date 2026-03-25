import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from pathlib import Path
from src.core.config import COMMUNITY_QUOTAS, COMMUNITY_PARTICIPANTS, CUPS_MAPPING
from src.core.database import load_from_supabase_db

# Paràmetres globals Subapp
TARGET_KWP = 83.6979816
PAVELLO_KWP = 127.2
TARGET_RATIO = TARGET_KWP / PAVELLO_KWP

def load_solar_curves():
    """Carrega les corbes de generació anual del Pavelló (127.2 kWp) i Sala Nova (17.1 kWp)."""
    # Usar camí relatiu a l'arrel del projecte (on estan els CSVs)
    # cle_optimizer.py està a src/ui/reports/, l'arrel és 3 nivells amunt
    base_path = Path(__file__).parents[3]
    
    # Pavello
    df_pav = pd.read_csv(base_path / "modelpavello.csv", sep=";", decimal=",")
    df_pav = df_pav.dropna(subset=['Produccio FV kWh']) # drop trailing empty rows if any
    
    # Sala nova
    df_sala = pd.read_csv(base_path / "modelsalanova.csv", sep=";", decimal=",")
    df_sala = df_sala.dropna(subset=['Produccio FV kWh'])
    
    # Garantir que tenen 8760 hores (un any no de traspàs)
    gen_pavello = df_pav['Produccio FV kWh'].values[:8760]
    gen_salanova = df_sala['Produccio FV kWh'].values[:8760]
    
    return gen_pavello, gen_salanova

@st.cache_data(ttl=3600)
def fetch_and_prep_consumption(year=2025):
    """Carrega el consum horari de l'any objectiu per als CUPS municipals."""
    # Primer, obtenim els anys disponibles de forma lleugera si cal, 
    # però per agilitzar, fem una crida específica per l'any demanat.
    start_str = f"{year}-01-01"
    end_str = f"{year}-12-31"
    
    # Carreguem només l'any que ens interessa per no topar amb el límit de 200k files
    df_raw = load_from_supabase_db(start_date=start_str, end_date=end_str)
    
    # Per saber els anys disponibles a tota la DB (pel selector), 
    # podríem fer una altra crida o heretar-ho. 
    # De moment, si no hi ha dades per aquest any, mirem si n'hi ha en general.
    if df_raw is None or df_raw.empty:
        # Consulta ràpida de tots els anys (això podria fallar pel límit, millor rpc)
        # Però per ara, si no hi ha 2025, retornem buit.
        return None, []
        
    df = df_raw.copy()
    available_years = [year] # Simplified since we filtered at source
    
    # Reconstruir el consum total (Xarxa + Autoconsum existent de Sala Nova)
    cups_data = {}
    municipal_cups_ids = [k for k, v in CUPS_MAPPING.items() if not str(v).startswith("Part. Privat")]
    for cups_id in municipal_cups_ids:
        cups_name = CUPS_MAPPING.get(cups_id)
        
        # Check if the friendly name is in the DB, fallback to ID just in case
        if cups_name in df.columns.get_level_values(0):
            target_col = cups_name
        elif cups_id in df.columns.get_level_values(0):
            target_col = cups_id
        else:
            continue
            
        cols = df[target_col].columns
        ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
        auto_col = [c for c in cols if 'AUTOCONS' in c]
        
        val_total = pd.Series(0.0, index=df.index)
        if ae_col:
            val_total = val_total.add(df[target_col][ae_col[0]], fill_value=0)
        if auto_col:
            val_total = val_total.add(df[target_col][auto_col[0]], fill_value=0)
            
        cups_data[cups_id] = val_total
        
    if not cups_data:
        return None, available_years
        
    df_final = pd.DataFrame(cups_data)
    full_index = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='h')
    df_final = df_final.groupby(df_final.index).mean()
    df_final = df_final.reindex(full_index, fill_value=0)
    
    return df_final, available_years

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
            'Coeficient Pavelló': coef_pav,
            'Consum Anual (kWh)': np.sum(consum),
            'Autoconsum SN (kWh)': total_auto_sn,
            'Autoconsum PAV (kWh)': total_auto_pav,
            'Autoconsum Total (kWh)': np.sum(autoconsum),
            'Excedents Compensats (kWh)': excedents_compensats_qty,
            'Excedents Llençats a la xarxa (kWh)': excedents_abocats_qty,
            'Estalvi Anual (€)': cost_nosolar_cups - cost_solar_cups,
            'Cobertura (%)': (np.sum(autoconsum) / np.sum(consum)) * 100 if np.sum(consum)>0 else 0,
            'Mensual': mensual_stats
        })
        
        total_cost_nosolar += cost_nosolar_cups
        total_cost_solar += cost_solar_cups
        
    total_savings = total_cost_nosolar - total_cost_solar
    return -total_savings, results_by_cups

def objective_function(coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price):
    savings_neg, _ = evaluate_coefficients(coefs, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price)
    return savings_neg

def run_optimization(df_consum, prices, excedent_price):
    cups_names = list(df_consum.columns)
    N = len(cups_names)
    gen_pavello, gen_salanova = load_solar_curves()
    
    # Valors inicials igualitaris
    init_guess = np.ones(N) * (TARGET_RATIO / N)
    
    # Límits: [0, 1] per cada coef
    bounds = [(0, 1) for _ in range(N)]
    
    # Restricció: suma de coeficients = TARGET_RATIO
    constraints = {'type': 'eq', 'fun': lambda c: np.sum(c) - TARGET_RATIO}
    
    with st.spinner("L'algoritme d'optimització iteratiu està determinant el repartiment màxim... (això pot trigar uns segons)"):
        opt_res = minimize(
            objective_function, 
            init_guess,
            args=(cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-4}
        )
        
    # Extraure resultats finals
    _, detailed_results = evaluate_coefficients(
        opt_res.x, cups_names, df_consum, gen_pavello, gen_salanova, prices, excedent_price
    )
    return detailed_results
    
@st.fragment
def render_cle_optimizer():
    st.subheader("⚙️ Optimitzador CLE: Assignació Repartiment Pavelló (RD 244/2019)")
    st.markdown("""
        Aquesta subaplicació determina automàticament els **coeficients fixes ideals anuals** per a la nova instal·lació **FV Pavelló (127.2 kWp)**, amb l'objectiu de distribuir exclusivament la quota reservada per a l'Ajuntament (83.69 kWp).
        
        **Criteris aplicats segons normativa ESP:**
        * L'energia provinent de la instal·lació Sala Nova es té en compte prèviament per calcular consums romanents.
        * Simulació amb freqüència **horària** utilitzant històrìcs de 2025 i extrapolació de les dues corbes FV.
        * El límit de compensació d'excedents mensual s'aplica estrictament abans d'impostos.
        * Totes les xifres econòmiques estan finalment gravades amb Impost Elèctric (5.11%) i IVA (21%).
    """)
    
    # UI Constants
    col1, col2, col3, col4, col5 = st.columns(5)
    p1 = col1.number_input("Preu P1 (€/kWh)", value=0.22, format="%.3f")
    p2 = col2.number_input("Preu P2 (€/kWh)", value=0.147, format="%.3f")
    p3 = col3.number_input("Preu P3 (€/kWh)", value=0.11, format="%.3f")
    p_exc = col4.number_input("Compensació Excedents (€)", value=0.07, format="%.3f")
    
    # Fem consulta ràpida per saber quins anys hi ha
    _, av_years = fetch_and_prep_consumption(2025)
    selected_year = col5.selectbox("Any d'Anàlisi", av_years if av_years else [2025], index=av_years.index(2025) if 2025 in av_years else 0)
    
    if st.button("Executar Motor d'Optimització", type="primary"):
        df_consum, years_found = fetch_and_prep_consumption(selected_year)
        if df_consum is None or df_consum.empty:
            st.error(f"No s'han trobat dades de consum suficients per als CUPS municipals per l'any {selected_year}.")
            if years_found:
                 st.info(f"Anys disponibles a la base de dades: {', '.join(map(str, years_found))}")
            else:
                 st.warning("La base de dades sembla estar buida o no s'ha pogut carregar.")
            return
            
        prices = calculate_tariffs(df_consum.index, p1, p2, p3)
        
        # Corre l'optimització
        detailed_results = run_optimization(df_consum, prices, p_exc)
        
        # --- PROCESSAMENT RESULTATS ---
        st.success("Optimització finalitzada amb èxit!")
        st.markdown("### Resultats d'Equilibri Resultants")
        
        df_res = pd.DataFrame(detailed_results)
        df_res = df_res.drop(columns=['Mensual'])
        
        # Format coeficient a 6 decimals
        df_res['Coeficient Pavelló'] = df_res['Coeficient Pavelló'].apply(lambda x: f"{x:.6f}")
        df_res['Estalvi Anual (€)'] = df_res['Estalvi Anual (€)'].apply(lambda x: f"{x:,.2f} €".replace(',','.'))
        df_res['Cobertura (%)'] = df_res['Cobertura (%)'].apply(lambda x: f"{x:.1f} %")
        
        df_res['Autoconsum (Sala Nova)'] = df_res['Autoconsum SN (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_res['Autoconsum (Pavelló)'] = df_res['Autoconsum PAV (kWh)'].apply(lambda x: f"{x:,.0f} kWh".replace(',','.'))
        df_res = df_res.drop(columns=['Autoconsum SN (kWh)', 'Autoconsum PAV (kWh)'])
        
        st.dataframe(df_res, use_container_width=True)
        
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

