import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import timedelta
from ...core.config import month_names, month_names_short, CUPS_MAPPING, COMMUNITY_PARTICIPANTS
from ...core.database import init_supabase, fetch_fv_data_chunked

def fetch_year_sums_rpc(year):
    """Fetches monthly sums for a specific year via RPC."""
    supabase = init_supabase()
    if not supabase: return []
    try:
        # The database function is get_monthly_sums(start_date, end_date)
        start_date = f"{year}-01-01T00:00:00Z"
        end_date = f"{year}-12-31T23:59:59Z"
        res = supabase.rpc("get_monthly_sums", {
            "start_date": start_date, 
            "end_date": end_date
        }).execute()
        return res.data if res.data else []
    except Exception as e:
        st.error(f"Error RPC sums ({year}): {e}")
        return []

def calculate_per_cup_totals_rpc(year_data):
    """Aggregates RPC data by CUPS Name. Output: Dict { 'CupName': total_kwh }"""
    if not year_data: return {}
    idf = pd.DataFrame(year_data)
    if idf.empty: return {}
    
    # Filter for AE_kWh and ignore AUTOCONS for base consumption ranking
    idf = idf[idf['cups'].str.contains('_AE_kWh') & ~idf['cups'].str.contains('AUTOCONS')]
    if idf.empty: return {}
    
    # Group by 'cups' key first (sum over months)
    grouped = idf.groupby('cups')['total_kwh'].sum()
    
    totals = {}
    for k, val in grouped.items():
        # k is like "Name___VAR" or "ID___VAR"
        name_part = str(k).split('___')[0] if '___' in str(k) else str(k).replace('_AE_kWh', '')
        # Map ID -> Name if it was an ID
        p_name = CUPS_MAPPING.get(name_part, name_part)
        totals[p_name] = totals.get(p_name, 0) + val
        
    return totals

def get_rpc_kpis(raw_data, subset_names):
    """Filters RPC data by a subset of CUPS and aggregates into (Total, MonthlySeries)."""
    if not raw_data: return 0, pd.Series(0.0, index=range(1, 13))
    df_r = pd.DataFrame(raw_data)
    if df_r.empty: return 0, pd.Series(0.0, index=range(1, 13))
    
    # Clean 'cups' field (it might be NAME___VARIABLE or ID___VARIABLE)
    df_r['clean_cups'] = df_r['cups'].apply(lambda x: str(x).split("___")[0] if "___" in str(x) else str(x).replace('_AE_kWh', ''))
    
    # Map any IDs in clean_cups to Names to match subset_names (which are Names)
    df_r['match_name'] = df_r['clean_cups'].apply(lambda x: CUPS_MAPPING.get(x, x))
    
    # Filter for Active Energy
    df_r = df_r[df_r['cups'].str.contains('_AE_kWh') & ~df_r['cups'].str.contains('AUTOCONS')]
    
    # Filter by subset labels
    target_names = set(subset_names)
    df_filtered = df_r[df_r['match_name'].isin(target_names)]
    
    if df_filtered.empty:
        return 0, pd.Series(0.0, index=range(1, 13))
        
    total_val = df_filtered['total_kwh'].sum()
    df_filtered = df_filtered.copy()
    df_filtered['m'] = pd.to_datetime(df_filtered['month']).dt.month
    monthly_s = df_filtered.groupby('m')['total_kwh'].sum()
    monthly_s = monthly_s.reindex(range(1, 13), fill_value=0.0)
    
    return total_val, monthly_s

def render_executive_report(df, lighting_cups, building_cups, all_cups, source_mode=""):
    """Renders the executive report tab content."""
    st.header("📊 Informe Executiu de Gestió Energètica")
    
    if source_mode != "Base de Dades (Supabase)":
        st.warning("⚠️ L'informe executiu està optimitzat per a dades de la Base de Dades. La comparativa amb anys anteriors podria no està disponible per a fitxers locals.")
    
    # 1. Year Selection
    today = datetime.date.today()
    all_years = []
    
    if source_mode == "Base de Dades (Supabase)":
        try:
            client = init_supabase()
            res = client.rpc("get_distinct_years").execute()
            all_years = sorted([int(item['year']) for item in res.data]) if res.data else []
        except:
            pass
            
    if not all_years:
        if df is not None:
            all_years = sorted(df.index.year.unique().tolist())
        else:
            all_years = [today.year - 1, today.year]

    if not all_years:
        st.warning("No hi ha dades disponibles per generar l'informe.")
        return

    col_y1, col_y2 = st.columns(2)
    target_year = col_y1.selectbox("Any de l'Informe", all_years, index=len(all_years)-1)
    
    prev_year_options = [y for y in all_years if y < target_year]
    if not prev_year_options:
        prev_year = target_year - 1
        col_y2.info(f"Comparant amb {prev_year} (per defecte)")
    else:
        prev_year = col_y2.selectbox("Any de Comparació", all_years if all_years else [target_year-1], index=max(0, all_years.index(target_year)-1) if target_year in all_years else 0)

    # 2. Data Fetching
    with st.spinner(f"Analitzant {target_year} vs {prev_year}..."):
        if source_mode == "Base de Dades (Supabase)":
            raw_target = fetch_year_sums_rpc(target_year)
            raw_prev = fetch_year_sums_rpc(prev_year)
            
            total_target, s_monthly_target = get_rpc_kpis(raw_target, all_cups)
            total_prev, s_monthly_prev = get_rpc_kpis(raw_prev, all_cups)
            
            light_target, _ = get_rpc_kpis(raw_target, lighting_cups)
            light_prev, _ = get_rpc_kpis(raw_prev, lighting_cups)
            
            build_target, _ = get_rpc_kpis(raw_target, building_cups)
            build_prev, _ = get_rpc_kpis(raw_prev, building_cups)
            
            per_cup_target = calculate_per_cup_totals_rpc(raw_target)
            per_cup_prev = calculate_per_cup_totals_rpc(raw_prev)
        else:
            # Local calculation logic (Simplified fallback)
            df_target = df[df.index.year == target_year] if df is not None else pd.DataFrame()
            df_prev = df[df.index.year == prev_year] if df is not None else pd.DataFrame()
            
            def get_sum_local(dframe, subset):
                if dframe.empty: return 0.0, pd.Series(0.0, index=range(1,13))
                # This is a bit complex for a local fallback in modular version, 
                # but let's keep it simple: sum all columns in subset
                # Filter AE cols
                target_cols = []
                for c in subset:
                    if c in dframe.columns.get_level_values(0):
                        sub_df = dframe[c]
                        ae_cols = [col for col in sub_df.columns if 'AE' in col and 'kWh' in col and 'AUTOCONS' not in col]
                        target_cols.extend([(c, ac) for ac in ae_cols])
                
                if not target_cols: return 0.0, pd.Series(0.0, index=range(1,13))
                
                total_s = dframe[target_cols].sum(axis=1)
                total_val = total_s.sum()
                monthly = total_s.groupby(total_s.index.month).sum().reindex(range(1,13), fill_value=0.0)
                return total_val, monthly

            total_target, s_monthly_target = get_sum_local(df_target, all_cups)
            total_prev, s_monthly_prev = get_sum_local(df_prev, all_cups)
            light_target, _ = get_sum_local(df_target, lighting_cups)
            light_prev, _ = get_sum_local(df_prev, lighting_cups)
            build_target, _ = get_sum_local(df_target, building_cups)
            build_prev, _ = get_sum_local(df_prev, building_cups)
            
            # Ranking dicts
            per_cup_target = {c: get_sum_local(df_target, [c])[0] for c in all_cups}
            per_cup_prev = {c: get_sum_local(df_prev, [c])[0] for c in all_cups}

    # --- 3. KPI SECTION ---
    st.subheader("💡 Indicadors Clau de Rendiment (KPIs)")
    c1, c2, c3 = st.columns(3)
    
    def delta_str(curr, prev):
        if prev == 0: return "n/a"
        diff = ((curr - prev) / prev) * 100
        return f"{diff:+.1f}%"
    
    with c1:
        st.metric(f"Consum Total {target_year}", f"{total_target:,.0f} kWh", delta_str(total_target, total_prev), delta_color="inverse")
    with c2:
        st.metric("📦 Enllumenat Públic", f"{light_target:,.0f} kWh", delta_str(light_target, light_prev), delta_color="inverse")
    with c3:
        st.metric("🏢 Edificis Municipals", f"{build_target:,.0f} kWh", delta_str(build_target, build_prev), delta_color="inverse")

    # --- 4. EVOLUTION CHART ---
    st.subheader(f"📈 Evolució Mensual: {target_year} vs {prev_year}")
    fig = go.Figure()
    
    months_label = [month_names.get(m, str(m)) for m in range(1, 13)]
    
    fig.add_trace(go.Bar(x=months_label, y=s_monthly_prev, name=f"Consum {prev_year}", marker_color='#BDC3C7'))
    fig.add_trace(go.Bar(x=months_label, y=s_monthly_target, name=f"Consum {target_year}", marker_color='#3498DB'))
    
    fig.update_layout(barmode='group', template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width='stretch')

    # --- 5. TOP MOVERS (RANKING) ---
    st.subheader("📉 Rànquing de Variacions (Per CUPS)")
    col_top1, col_top2 = st.columns(2)
    
    diffs = []
    for cup_name in all_cups:
        val_t = per_cup_target.get(cup_name, 0)
        val_p = per_cup_prev.get(cup_name, 0)
        diff = val_t - val_p
        pct = (diff / val_p * 100) if val_p > 0 else 0
        diffs.append({"Nom": cup_name, "Diferència (kWh)": diff, "Diferència (%)": pct})
    
    df_diffs = pd.DataFrame(diffs)
    
    with col_top1:
        st.markdown("##### 📉 Top 5 Estalvis")
        savings = df_diffs[df_diffs["Diferència (kWh)"] < 0].sort_values("Diferència (kWh)", ascending=True).head(5)
        if not savings.empty:
             st.table(savings.style.format({"Diferència (kWh)": "{:,.0f}", "Diferència (%)": "{:+.1f}%"}))
        else:
             st.info("Sense estalvis significatius.")
             
    with col_top2:
        st.markdown("##### 📈 Top 5 Augments")
        increases = df_diffs[df_diffs["Diferència (kWh)"] > 0].sort_values("Diferència (kWh)", ascending=False).head(5)
        if not increases.empty:
             st.table(increases.style.format({"Diferència (kWh)": "{:,.0f}", "Diferència (%)": "{:+.1f}%"}))
        else:
             st.info("Sense augments significatius.")

    # --- 6. MONTHLY DETAIL TABLE ---
    with st.expander("📅 Veure Detall Mensual en Taula"):
        df_table = pd.DataFrame({
            "Mes": months_label,
            f"{prev_year} (kWh)": [s_monthly_prev.get(m, 0) for m in range(1, 13)],
            f"{target_year} (kWh)": [s_monthly_target.get(m, 0) for m in range(1, 13)]
        })
        st.dataframe(df_table.style.format({f"{prev_year} (kWh)": "{:,.0f}", f"{target_year} (kWh)": "{:,.0f}"}), width='stretch', hide_index=True)

    # --- 7. ENERGY COMMUNITY IMPACT ---
    st.markdown("---")
    st.subheader("☀️ Impacte Comunitat Energètica Local")
    
    # Identify participants from whitelist
    clean_whitelist = [x.strip().upper() for x in COMMUNITY_PARTICIPANTS]
    rev_map = {v: k for k, v in CUPS_MAPPING.items()}
    
    self_cups = []
    for c in all_cups:
        cid = rev_map.get(c, c)
        if str(cid).strip().upper() in clean_whitelist:
            self_cups.append(c)
            
    if not self_cups:
        st.info("Cap CUPS de la Comunitat Energètica detectat en la selecció actual.")
    else:
        # Calculate Monthly Autoconsum
        if source_mode == "Base de Dades (Supabase)":
            def sum_autocons_rpc(rpc_data, allowed_names):
                if not rpc_data: return 0.0, pd.Series(0.0, index=range(1, 13))
                df_r = pd.DataFrame(rpc_data)
                df_auto = df_r[df_r['cups'].str.contains('AUTOCONS')]
                if df_auto.empty: return 0.0, pd.Series(0.0, index=range(1, 13))
                
                # Filter by names
                def is_match(k):
                    name = k.split("___")[0] if "___" in k else k
                    return name in allowed_names
                
                df_auto = df_auto[df_auto['cups'].apply(is_match)]
                if df_auto.empty: return 0.0, pd.Series(0.0, index=range(1, 13))
                
                tot = df_auto['total_kwh'].sum()
                df_auto['m'] = pd.to_datetime(df_auto['month']).dt.month
                ser = df_auto.groupby('m')['total_kwh'].sum().reindex(range(1, 13), fill_value=0.0)
                return tot, ser

            total_self, s_monthly_self = sum_autocons_rpc(raw_target, set(self_cups))
        else:
            # Local
            s_monthly_self = pd.Series(0.0, index=range(1, 13))
            total_self = 0
            if df is not None:
                df_target = df[df.index.year == target_year]
                for c in self_cups:
                    if c in df_target.columns:
                        auto_cols = [col for col in df_target[c].columns if 'AUTOCONS' in col]
                        if auto_cols:
                            total_self += df_target[c][auto_cols[0]].sum()
                            m_sums = df_target[c][auto_cols[0]].groupby(df_target.index.month).sum()
                            for m, val in m_sums.items():
                                s_monthly_self[m] += val

        # Chart: Comparative Autoconsum vs Total
        fig_comm = go.Figure()
        # Adjusted Grid (Total - Self)
        net_grid = [max(0, s_monthly_target.get(m, 0) - s_monthly_self.get(m, 0)) for m in range(1, 13)]
        
        fig_comm.add_trace(go.Bar(x=months_label, y=s_monthly_self, name="Autoconsum", marker_color="gold"))
        fig_comm.add_trace(go.Bar(x=months_label, y=net_grid, name="Xarxa (Consum Net)", marker_color="gray"))
        
        fig_comm.update_layout(barmode='stack', title="Cobertura d'Autoconsum vs Consum Total", template="plotly_white")
        st.plotly_chart(fig_comm, width='stretch')
        
        # Details Table
        table_data = []
        for i, m in enumerate(range(1, 13)):
            t = s_monthly_target.get(m, 0)
            s = s_monthly_self.get(m, 0)
            pct = (s / t * 100) if t > 0 else 0
            table_data.append({"Mes": months_label[i], "Consum Total (kWh)": t, "Autoconsum (kWh)": s, "% Autoconsum": pct})
        
        # Total row
        t_total = s_monthly_target.sum()
        s_total = s_monthly_self.sum()
        pct_total = (s_total / t_total * 100) if t_total > 0 else 0
        table_data.append({"Mes": "**TOTAL ANY**", "Consum Total (kWh)": t_total, "Autoconsum (kWh)": s_total, "% Autoconsum": pct_total})
        
        st.table(pd.DataFrame(table_data).style.format({"Consum Total (kWh)": "{:,.0f}", "Autoconsum (kWh)": "{:,.0f}", "% Autoconsum": "{:.1f}%"}))

    # --- 8. SALA NOVA PV PRODUCTION ---
    st.markdown("---")
    st.header("☀️ Producció Fotovoltaica Sala Nova")
    try:
        data_fv = fetch_fv_data_chunked()
        if data_fv:
            df_fv = pd.DataFrame(data_fv)
            df_fv['reading_time'] = pd.to_datetime(df_fv['reading_time'])
            df_fv_target = df_fv[df_fv['reading_time'].dt.year == target_year]
            df_fv_prev = df_fv[df_fv['reading_time'].dt.year == prev_year]
            
            if not df_fv_target.empty:
                gen_target = df_fv_target['potencia_fv'].sum()
                gen_prev = df_fv_prev['potencia_fv'].sum() if not df_fv_prev.empty else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Producció {target_year}", f"{gen_target:,.0f} kWh", delta_str(gen_target, gen_prev))
                
                # Monthly split
                m_gen_target = df_fv_target.groupby(df_fv_target['reading_time'].dt.month)['potencia_fv'].sum().reindex(range(1,13), fill_value=0.0)
                m_gen_prev = df_fv_prev.groupby(df_fv_prev['reading_time'].dt.month)['potencia_fv'].sum().reindex(range(1,13), fill_value=0.0)
                
                best_m = m_gen_target.idxmax()
                c2.metric("Millor Mes", month_names.get(best_m, str(best_m)), f"{m_gen_target.max():,.0f} kWh")
                
                # Estimated savings (using a generic 0.15 €/kWh or similar)
                c3.metric("Estalvi Estimat", f"{gen_target * 0.15:,.2f} €", help="Càlcul basat en un preu mitjà de 0.15 €/kWh")
                
                # Comparison Chart
                fig_fv = go.Figure()
                fig_fv.add_trace(go.Bar(x=months_label, y=m_gen_prev, name=f"Producció {prev_year}", marker_color="#E5E7E9"))
                fig_fv.add_trace(go.Bar(x=months_label, y=m_gen_target, name=f"Producció {target_year}", marker_color="#F1C40F"))
                fig_fv.update_layout(title="Comparativa Producció Fotovoltaica Mensual", template="plotly_white")
                st.plotly_chart(fig_fv, width='stretch')
            else:
                st.info(f"No hi ha dades de la Sala Nova per a l'any {target_year}.")
        else:
            st.info("No s'han trobat dades de producció fotovoltaica externa.")
    except Exception as e:
        st.error(f"Error carregant dades FV: {e}")
