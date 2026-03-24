import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from astral import LocationInfo
from astral.sun import sun
import datetime
from datetime import timedelta
import numpy as np
import time

# --- MODULAR IMPORTS ---
from src.core.config import CUPS_MAPPING, COMMUNITY_PARTICIPANTS, month_names, month_names_short, PAGE_TITLE, PAGE_ICON, apply_custom_styles
from src.core.database import init_supabase, load_fv_sala_nova_data, load_from_supabase_db, sync_csv_to_db
from src.ui.layout import render_login, render_sidebar, init_session_state
from src.ui.views.executive import render_executive_report
from src.ui.reports.cle_optimizer import render_cle_optimizer
from src.utils.parsers import parse_processed_csv, process_edistribucion_files
from src.utils.data_utils import classify_cups_by_name, detect_self_consumption_cups, get_date_range, shift_date

# Page Config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout='wide',
    initial_sidebar_state='expanded'
)

# Apply CSS
apply_custom_styles()

# --- Main App Interface ---

def main():
    st.title('Comptabilitat elèctrica Ajuntament de Sant Jordi Desvalls')
    
    # Init Session State
    init_session_state()
    
    # --- LOGIN FLOW ---
    if not st.session_state.user:
        render_login()
        return
    # ------------------
    
    # Sidebar & Data Source
    # --- Mode Selection (Moved to Top for Context) ---
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Mode de Visualització", ["Expert", "Informe Executiu"], index=1)
    
    # Render Sidebar (User info, Logout, and Source if Expert)
    source_mode = render_sidebar(show_source_selector=(app_mode == 'Expert'))
    if not source_mode: return
    
    df = None
    
    if source_mode == "Pujar CSV Local (Processat)":
        uploaded_file = st.sidebar.file_uploader("Pujar CSV (Format Horari)", type=["csv"], help="Format: Datetime Index, Columnes=CUPS")
        if uploaded_file is not None:
             with st.spinner('Processant CSV...'):
                 df = parse_processed_csv(uploaded_file)
             
             if df is not None:
                 st.write("---")
                 st.markdown("##### ☁️ Configuració de Càrrega")
                 upload_mode = st.radio(
                     "Mode de Sincronització", 
                     ["Fusionar / Actualitzar", "⚠️ Esborrar Tot i Reemplaçar"],
                     help="Fusionar: Actualitza dades existents i afegeix les noves. \nEsborrar: Elimina TOTA la base de dades abans de carregar aquest fitxer."
                 )
                 
                 if st.button("💾 Guardar a Base de Dades"):
                     mode_code = "replace" if "Esborrar" in upload_mode else "merge"
                     sync_csv_to_db(df, mode=mode_code)
                     
    elif source_mode == "Importar Edistribucion (Originals)":
        st.sidebar.info("Puja fitxers originals (.csv) d'Edistribucion. S'agruparan i normalitzaran automàticament.")
        uploaded_files = st.sidebar.file_uploader("Pujar CSVs Originals", type=["csv"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.sidebar.button("⚙️ Processar i Previsualitzar"):
                 with st.spinner("Processant i fusionant fitxers..."):
                     df = process_edistribucion_files(uploaded_files)
                     if df is not None:
                         st.success(f"Processats {len(uploaded_files)} fitxers correctament!")
            
            if not st.session_state.get('edist_processed', False):
                 with st.spinner("Llegint fitxers..."):
                     df = process_edistribucion_files(uploaded_files)
            
            if df is not None:
                 st.write("---")
                 st.markdown("##### ☁️ Configuració de Càrrega")
                 st.info(f"Dades preparades: {len(df)} hores x {len(df.columns)//2} CUPS")
                 
                 upload_mode = st.radio(
                     "Mode de Sincronització", 
                     ["Fusionar / Actualitzar", "⚠️ Esborrar Tot i Reemplaçar"],
                     key="upload_mode_edist",
                     help="Fusionar: Recomanat per afegir nous mesos."
                 )
                 
                 if st.button("💾 Guardar a Base de Dades (Originals)"):
                     mode_code = "replace" if "Esborrar" in upload_mode else "merge"
                     sync_csv_to_db(df, mode=mode_code)

    else: # Base de Dades (Supabase)
        st.sidebar.markdown("---")
        today = datetime.date.today()
        
        try:
             client = init_supabase()
             res = client.rpc("get_distinct_years").execute()
             db_years = [item['year'] for item in res.data] if res.data else []
             max_year_db = max(db_years) if db_years else today.year
             min_year_db = min(db_years) if db_years else today.year
        except:
             max_year_db = today.year
             min_year_db = today.year
             
        default_start = datetime.date(min_year_db, 1, 1)
        default_end = datetime.date(max_year_db, 12, 31)
        
        db_start = default_start
        db_end = default_end
        
        if app_mode == 'Expert':
            with st.sidebar.expander("📅 Rang de Dades (DB)", expanded=False):
             db_start = st.date_input("Inici", default_start)
             db_end = st.date_input("Fi", default_end)
             reload_btn = st.button("🔄 Actualitzar Dades")
             
        with st.spinner(f"Carregant dades ({db_start} - {db_end})..."):
             df = load_from_supabase_db(start_date=db_start, end_date=db_end)

        if df is None:
            st.info("La base de dades està buida o no s'han trobat dades en aquest rang.")
            st.info("Utilitza 'Pujar CSV Local' per carregar les primeres dades.")

    if df is not None:
        # Standardize Index (ensure Datetime)
        if not isinstance(df.index, pd.DatetimeIndex):
             # Try to recover index if load_from_db didn't set it perfectly or parse_data variation
             pass # Logic handles it inside helpers usually

        # Init anchor from Data
        min_csv_date = df.index.min().date()
        max_csv_date = df.index.max().date()
            
        # If anchor is out of range, default to LATEST date (Max) but warn
        if not (min_csv_date <= st.session_state.anchor_date <= max_csv_date):
             st.toast(f"Data ajustada al rang disponible: {min_csv_date} - {max_csv_date}", icon="⚠️")
             # Only adjust if it's REALLY far off (e.g. wrong year entirely when we have multiyer)
             if st.session_state.anchor_date > max_csv_date:
                 st.session_state.anchor_date = max_csv_date
             elif st.session_state.anchor_date < min_csv_date:
                 st.session_state.anchor_date = min_csv_date

        # --- Classification Step ---
        st.subheader("🤖 Classificació Automàtica de CUPS")
        lighting_cups, building_cups = classify_cups_by_name(df)
        # Identify self-consumers from Whitelist directly
        # DF columns are Names, Whitelist is IDs. Reverse map needed.
        rev_map_local = {v: k for k, v in CUPS_MAPPING.items()}
        clean_whitelist = [x.strip().upper() for x in COMMUNITY_PARTICIPANTS]
        
        all_cols_idx = df.columns.get_level_values(0).unique()
        self_consumption_cups = []
        for c in all_cols_idx:
            cid = rev_map_local.get(c, c)
            if str(cid).strip().upper() in clean_whitelist:
                self_consumption_cups.append(c)
        
        all_cups = df.columns.get_level_values(0).unique().tolist()
        
        # Show Classification Logic Results
        with st.expander("Veure Detall Classificació", expanded=True):
            col_class_1, col_class_2, col_class_3 = st.columns(3)
            
            rev_map = {v: k for k, v in CUPS_MAPPING.items()}
            def make_display_df(items):
                rows = []
                for it in items:
                    rows.append({"Nom": it, "CUPS": rev_map.get(it, it)})
                return pd.DataFrame(rows)
            
            with col_class_1:
                st.markdown(f"**💡 Enllumenat ({len(lighting_cups)})**")
                st.dataframe(make_display_df(lighting_cups), hide_index=True)
            with col_class_2:
                st.markdown(f"**🏢 Edificis ({len(building_cups)})**")
                st.dataframe(make_display_df(building_cups), hide_index=True)
            with col_class_3:
                st.markdown(f"**☀️ Autoconsum ({len(self_consumption_cups)})**")
                if self_consumption_cups:
                    st.dataframe(make_display_df(self_consumption_cups), hide_index=True)
                else:
                    st.info("Cap detectat.")
        
        # --- Mode Selection (Already top) ---
        # st.sidebar.divider()
        # app_mode = st.sidebar.radio("Mode de Visualització", ["Expert", "Informe Executiu"], index=1)

        if app_mode == "Informe Executiu":
            render_executive_report(df, lighting_cups, building_cups, all_cups, source_mode=source_mode)
            return 

        # === MODE: EXPERT (Implicit) ===
        
        # --- Global Filter Logic Helpers ---
        def set_cups_selection(group_type):
            if group_type == 'All':
                st.session_state.selected_cups_list = all_cups
            elif group_type == 'Lighting':
                st.session_state.selected_cups_list = lighting_cups
            elif group_type == 'Buildings':
                st.session_state.selected_cups_list = building_cups
            elif group_type == 'Solar':
                st.session_state.selected_cups_list = self_consumption_cups
            else:
                st.session_state.selected_cups_list = []

        # Initialize selection if empty on first load
        if not st.session_state.selected_cups_list:
             st.session_state.selected_cups_list = all_cups

        # --- Tabs ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Panell Global", "📈 Comparativa", "🌃 Auditor Enllumenat", "🤖 AI Advisor", "☀️ Autoconsum", "☀️ FV Sala Nova", "⚙️ Optimitzador CLE"])
        
        @st.fragment
        def render_tab_global(df_data, cups_list):
            # 1. Controls Row
            col_nav1, col_nav2, col_nav3 = st.columns([2, 1, 3])
            
            with col_nav1:
                 mode = st.selectbox("Escala Temporal", ["Diària", "Setmanal", "Mensual", "Anual"], index=3, key="view_mode_t1_frag")
            
            with col_nav2:
                col_b1, col_b2 = st.columns(2)
                if col_b1.button("⬅️", key="prev_t1_frag"):
                    st.session_state.anchor_date = shift_date(mode, st.session_state.anchor_date, -1)
                if col_b2.button("➡️", key="next_t1_frag"):
                    st.session_state.anchor_date = shift_date(mode, st.session_state.anchor_date, 1)

            start_d, end_d, freq_alias = get_date_range(mode, st.session_state.anchor_date)
            
            with col_nav3:
                st.subheader(f"📅 {start_d} - {end_d}")

            # 2. Filter Data based on time
            mask_time = (df_data.index.date >= start_d) & (df_data.index.date <= end_d)
            df_filtered_t1 = df_data.loc[mask_time]
            
            selected_cups_t1 = st.multiselect("Filtrar CUPS (Agregat)", cups_list, default=cups_list, key="sel_cups_t1_frag")
            
            # Aggregation
            def get_aggregated_data(data_df, c_list):
                total_ae = pd.Series(0, index=data_df.index)
                total_autocons = pd.Series(0, index=data_df.index)
                for cups in c_list:
                    if cups not in data_df.columns.get_level_values(0): continue
                    cols = data_df[cups].columns
                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                    autocons_col = [c for c in cols if 'AUTOCONS' in c]
                    if ae_col: total_ae = total_ae.add(data_df[cups][ae_col[0]], fill_value=0)
                    if autocons_col: total_autocons = total_autocons.add(data_df[cups][autocons_col[0]], fill_value=0)
                return total_ae, total_autocons

            agg_ae, agg_autocons = get_aggregated_data(df_filtered_t1, selected_cups_t1)
            
            # 3. Display KPIs & Chart
            if not agg_ae.empty and agg_ae.sum() > 0:
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Consum Xarxa", f"{agg_ae.sum():,.2f} kWh")
                kpi2.metric("Autoconsum", f"{agg_autocons.sum():,.2f} kWh")
                total_demand = agg_ae.sum() + agg_autocons.sum()
                kpi3.metric("Autosuficiència", f"{(agg_autocons.sum()/total_demand*100 if total_demand > 0 else 0):.2f} %")
                
                chart_ae = agg_ae
                chart_ac = agg_autocons
                
                if freq_alias != 'h': # Not raw
                    chart_ae = agg_ae.resample(freq_alias).sum()
                    chart_ac = agg_autocons.resample(freq_alias).sum()
                    
                full_idx = None
                if mode == 'Anual':
                     full_idx = pd.date_range(start=start_d, end=end_d, freq='ME') 
                elif mode == 'Mensual' or mode == 'Setmanal':
                     full_idx = pd.date_range(start=start_d, end=end_d, freq='D')
                elif mode == 'Diària':
                     full_idx = pd.date_range(start=start_d, periods=24, freq='h')
                
                if full_idx is not None:
                     chart_ae = chart_ae.reindex(full_idx, fill_value=0)
                     chart_ac = chart_ac.reindex(full_idx, fill_value=0)

                fig_line = go.Figure()
                fig_line.add_trace(go.Bar(x=chart_ae.index, y=chart_ae, name='Consum Xarxa', marker_color='#EF553B') if freq_alias != '1h' else go.Scatter(x=chart_ae.index, y=chart_ae, name='Consum Xarxa', line=dict(color='#EF553B'), fill='tozeroy'))
                
                if agg_autocons.sum() > 0:
                     fig_line.add_trace(go.Bar(x=chart_ac.index, y=chart_ac, name='Autoconsum', marker_color='#00CC96') if freq_alias != '1h' else go.Scatter(x=chart_ac.index, y=chart_ac, name='Autoconsum', line=dict(color='#00CC96'), fill='tozeroy'))

                xaxis_args = {'title': "Temps"}
                if mode == 'Diària':
                    xaxis_args['dtick'] = 3600000 * 1 
                    xaxis_args['tickformat'] = "%H:%M"
                elif mode == 'Setmanal':
                    xaxis_args['dtick'] = 86400000.0 
                    xaxis_args['tickformat'] = "%d/%m"
                elif mode == 'Mensual':
                     xaxis_args['dtick'] = 86400000.0 
                     xaxis_args['tickformat'] = "%d"
                elif mode == 'Anual':
                     xaxis_args['dtick'] = "M1" 
                     xaxis_args['tickformat'] = "%b"

                fig_line.update_layout(title="Evolució del Consum", barmode='stack', hovermode="x unified", xaxis=xaxis_args)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("Sense dades per aquest període/selecció.")

        # === TAB 1: Global Dashboard ===
        with tab1:
            st.header("Visió General")
            render_tab_global(df, all_cups)


        @st.fragment
        def render_tab_comparative(df_data, cups_list, lighting_c, building_c):
            # 1. Quick Selectors
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            
            def update_selection(new_list):
                st.session_state["t2_multi_comp"] = new_list
            
            if col_btn1.button("Tots (All)", key="t2_btn_all"): 
                update_selection(cups_list)
            if col_btn2.button("Enllumenat", key="t2_btn_light"): 
                update_selection(lighting_c)
            if col_btn3.button("Edificis (Resta)", key="t2_btn_build"): 
                update_selection(building_c)
            if col_btn4.button("Netejar", key="t2_btn_clear"): 
                update_selection([])
            
            if "t2_multi_comp" not in st.session_state:
                 st.session_state["t2_multi_comp"] = cups_list
            
            current_selection = st.multiselect("CUPS Seleccionats", cups_list, key="t2_multi_comp")
            
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns([2, 1, 3])
            with col_c1:
                mode_t2 = st.selectbox("Escala Temporal (Comparativa)", ["Diària", "Setmanal", "Mensual", "Anual"], index=3, key="mode_t2_frag")
            with col_c2:
                col_cb1, col_cb2 = st.columns(2)
                if col_cb1.button("⬅️", key="prev_t2_frag"):
                    st.session_state.anchor_date = shift_date(mode_t2, st.session_state.anchor_date, -1)
                if col_cb2.button("➡️", key="next_t2_frag"):
                     st.session_state.anchor_date = shift_date(mode_t2, st.session_state.anchor_date, 1)
            
            start_d2, end_d2, freq_alias2 = get_date_range(mode_t2, st.session_state.anchor_date)
             
            with col_c3:
                 st.subheader(f"📅 {start_d2} - {end_d2}")
                 
            # 3. Bar Chart (Stacked by CUPS)
            mask_time2 = (df_data.index.date >= start_d2) & (df_data.index.date <= end_d2)
            df_filtered_t2 = df_data.loc[mask_time2]
            
            if not df_filtered_t2.empty and current_selection:
                plot_data = []
                for cup in current_selection:
                    if cup not in df_filtered_t2.columns.get_level_values(0): continue
                    cols = df_filtered_t2[cup].columns
                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                    
                    if ae_col:
                        series = df_filtered_t2[cup][ae_col[0]]
                        if freq_alias2 != 'h':
                            series = series.resample(freq_alias2).sum()
                        
                        tmp = series.reset_index()
                        tmp.columns = ['Datetime', 'kWh']
                        tmp['CUPS'] = cup
                        plot_data.append(tmp)
                
                if plot_data:
                    final_plot_df = pd.concat(plot_data)
                    fig_bar = px.bar(
                        final_plot_df, x='Datetime', y='kWh', color='CUPS', 
                        title=f"Consum Desglossat ({mode_t2})",
                        text_auto='.2s' if len(current_selection) < 5 else False
                    )
                    
                    xaxis_args2 = {'title': "Temps"}
                    if mode_t2 == 'Diària':
                        xaxis_args2['dtick'] = 3600000 * 1
                        xaxis_args2['tickformat'] = "%H:%M"
                    elif mode_t2 == 'Setmanal':
                        xaxis_args2['dtick'] = 86400000.0
                        xaxis_args2['tickformat'] = "%d/%m"
                    elif mode_t2 == 'Mensual':
                        xaxis_args2['dtick'] = 86400000.0
                        xaxis_args2['tickformat'] = "%d"
                    elif mode_t2 == 'Anual':
                        xaxis_args2['dtick'] = "M1"
                        xaxis_args2['tickformat'] = "%b"
                        
                    fig_bar.update_layout(xaxis=xaxis_args2)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("No s'han trobat dades de consum per als CUPS seleccionats.")
            else:
                st.info("Selecciona rang i CUPS per visualitzar.")
                
            # 4. Flexible Comparison Chart
            st.subheader("Comparativa de Sèries (Tendències)")
            col_sel, col_res = st.columns([4, 1])
            comp_options = ["TOTAL", "ENLLUMENAT (Agregat)", "EDIFICIS (Agregat)"] + sorted(cups_list)
            
            if "comp_series_sel" not in st.session_state:
                st.session_state.comp_series_sel = ["TOTAL"]
                
            if col_res.button("Restablir (Total)", key="btn_reset_comp_frag"):
                st.session_state.comp_series_sel = ["TOTAL"]
            
            selection_series = col_sel.multiselect("Afegir/Treure Sèries al Gràfic", comp_options, key="series_multiselect_frag", default=st.session_state.comp_series_sel)
            
            available_years = sorted(df_data.index.year.unique())
            col_y_sel, _ = st.columns([2, 3])
            selected_years_comp = col_y_sel.multiselect("Seleccionar Anys a Comparar", available_years, default=available_years, key="years_comp_sel_frag")

            # Helper specifically for comparative fallback mode
            def get_aggregated_data_comp(data_df, c_list):
                total_ae = pd.Series(0, index=data_df.index)
                for cups in c_list:
                    if cups not in data_df.columns.get_level_values(0): continue
                    cols = data_df[cups].columns
                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                    if ae_col: total_ae = total_ae.add(data_df[cups][ae_col[0]], fill_value=0)
                return total_ae, None

            if selection_series:
                fig_comp = go.Figure()
                current_anchor = st.session_state.anchor_date
                
                if not selected_years_comp:
                    st.warning("Selecciona almenys un any per comparar.")
                else:
                    for item in selection_series:
                        if item == "TOTAL":
                            s_full, _ = get_aggregated_data_comp(df_data, cups_list)
                        elif item == "ENLLUMENAT (Agregat)":
                            s_full, _ = get_aggregated_data_comp(df_data, lighting_c)
                        elif item == "EDIFICIS (Agregat)":
                            s_full, _ = get_aggregated_data_comp(df_data, building_c)
                        else:
                            if item in cups_list and item in df_data.columns.get_level_values(0):
                                cols = df_data[item].columns
                                ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                                if ae_col:
                                    s_full = df_data[item][ae_col[0]]
                                else:
                                    continue
                            else: continue
                        
                        if mode_t2 == 'Anual':
                            s_monthly = s_full.resample('ME').sum()
                            month_map = {1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                                         7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Oct', 11: 'Nov', 12: 'Des'}
                            for y in selected_years_comp:
                                s_year = s_monthly[s_monthly.index.year == y]
                                if not s_year.empty:
                                    x_vals = s_year.index.month
                                    fig_comp.add_trace(go.Scatter(
                                        x=x_vals, y=s_year, name=f"{item} ({y})", mode='lines+markers',
                                        hovertemplate="%{y:.2f} kWh<extra></extra>"
                                    ))
                            xaxis_args_comp = {
                                'title': "Mes", 'tickmode': 'array',
                                'tickvals': list(range(1, 13)), 'ticktext': list(month_map.values())
                            }

                        elif mode_t2 == 'Mensual':
                            target_month = current_anchor.month
                            target_month_name = current_anchor.strftime("%B")
                            s_daily = s_full.resample('1d').sum()
                            for y in selected_years_comp:
                                mask = (s_daily.index.year == y) & (s_daily.index.month == target_month)
                                s_sub = s_daily[mask]
                                if not s_sub.empty:
                                    x_vals = s_sub.index.day
                                    fig_comp.add_trace(go.Scatter(x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'))
                            xaxis_args_comp = {'title': f"Dia ({target_month_name})", 'dtick': 1}

                        elif mode_t2 == 'Setmanal':
                            target_week = current_anchor.isocalendar().week
                            s_daily = s_full.resample('1d').sum()
                            for y in selected_years_comp:
                                mask = (s_daily.index.isocalendar().week == target_week) & (s_daily.index.year == y)
                                s_sub = s_daily[mask]
                                if not s_sub.empty:
                                    x_vals = s_sub.index.strftime('%a')
                                    fig_comp.add_trace(go.Scatter(x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'))
                            xaxis_args_comp = {'title': f"Dia de la Setmana {target_week}", 'dtick': "M1"}

                        elif mode_t2 == 'Diària':
                            target_day = current_anchor.day
                            target_month = current_anchor.month
                            s_hourly = s_full
                            for y in selected_years_comp:
                                mask = (s_hourly.index.year == y) & (s_hourly.index.month == target_month) & (s_hourly.index.day == target_day)
                                s_sub = s_hourly[mask]
                                if not s_sub.empty:
                                    x_vals = s_sub.index.hour
                                    fig_comp.add_trace(go.Scatter(x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'))
                            xaxis_args_comp = {'title': f"Hora ({target_day}/{target_month})", 'tickmode': 'linear', 'dtick': 1}

                    fig_comp.update_layout(title="Evolució Comparativa (Multianual)", xaxis=xaxis_args_comp, hovermode="x unified")
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # SUMMARY TABLE
                    st.subheader("Resum Series Seleccionades (Període/Anys)")
                    if selected_years_comp:
                        data_matrix = []
                        for it in selection_series:
                            row_dict = {"Sèrie": it}
                            if it == "TOTAL": s_f, _ = get_aggregated_data_comp(df_data, cups_list)
                            elif it == "ENLLUMENAT (Agregat)": s_f, _ = get_aggregated_data_comp(df_data, lighting_c)
                            elif it == "EDIFICIS (Agregat)": s_f, _ = get_aggregated_data_comp(df_data, building_c)
                            else:
                                 if it in df_data.columns.get_level_values(0):
                                     cols = df_data[it].columns
                                     ae_c = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                                     if ae_c: s_f = df_data[it][ae_c[0]]
                                     else: continue
                                 else: continue
                            
                            total_row = 0
                            for y in selected_years_comp:
                                mask_y = s_f.index.year == y
                                val_y = s_f[mask_y].sum()
                                row_dict[str(y)] = f"{val_y:.2f}"
                                total_row += val_y
                            
                            data_matrix.append(row_dict)
                        
                        st.dataframe(pd.DataFrame(data_matrix))
                        
                        # Bar Chart Visualization
                        st.caption("Visualització Gràfica del Resum Anual")
                        fig_summ_bar = go.Figure()
                        for y in selected_years_comp:
                            y_values = []
                            for row in data_matrix:
                                val_str = row.get(str(y), "0")
                                y_values.append(float(val_str))
                            
                            fig_summ_bar.add_trace(go.Bar(
                                name=str(y),
                                x=[r["Sèrie"] for r in data_matrix],
                                y=y_values
                            ))
                            
                        fig_summ_bar.update_layout(barmode='group', template="plotly_white", margin=dict(t=10, b=10))
                        st.plotly_chart(fig_summ_bar, use_container_width=True)

        # === TAB 2: Comparative Analysis ===
        with tab2:
            st.header("Anàlisi Comparatiu")
            render_tab_comparative(df, all_cups, lighting_cups, building_cups)

        # --- Tab 3: Public Lighting Auditor ---
        with tab3:
            st.header("Auditoria d'Enllumenat Públic")
            
            lighting_selected = st.multiselect("Seleccionar Enllumenat a Auditar", lighting_cups, default=lighting_cups)
            
            if not lighting_selected:
                st.warning("No hi ha cap CUPS d'enllumenat seleccionat.")
            else:
                # 1. Automated Scanner
                st.subheader("🕵️ Detecció d'Incidències (Massiva)")
                
                # Year Selector for Scanner
                available_years_scan = sorted(df.index.year.unique())
                scan_years = st.multiselect("Anys a Escanejar", available_years_scan, default=available_years_scan, key="scan_years_sel")
                
                if st.button("Buscar Anomalies"):
                    should_run = True
                else:
                    # Logic to Auto-Run if years changed and we have results
                    if 'scan_years_last_run' in st.session_state:
                        if set(st.session_state.scan_years_last_run) != set(scan_years):
                            should_run = True
                        else:
                            should_run = False
                    else:
                        should_run = False
                
                # RUN SCANNER
                if should_run:
                    if not scan_years:
                        st.warning("Selecciona almenys un any per escanejar.")
                    else:
                        with st.spinner(f"Analitzant CUPS per als anys: {scan_years}..."):
                            anomalies = []
                            total_excess_kwh = 0.0
                            
                            # Filter DF per Selected Years
                            df_scan = df[df.index.year.isin(scan_years)]
                            
                            # Optimize: Iterate by day is slow if many cups. 
                            # But logic requires sunset/sunrise which is daily.
                            unique_days_scan = sorted(list(set(df_scan.index.date)))
                            
                            progress_bar = st.progress(0)
                            total_steps = len(unique_days_scan)
                            
                            for i, d in enumerate(unique_days_scan):
                                # Sun info
                                city = LocationInfo("Girona", "Catalonia", "Europe/Madrid", 41.9, 2.8)
                                s = sun(city.observer, date=d)
                                sunrise = s['sunrise'].replace(tzinfo=None)
                                sunset = s['sunset'].replace(tzinfo=None)
                                
                                # Get day data slice
                                day_data = df_scan.loc[df_scan.index.date == d]
                                
                                for cup in lighting_selected:
                                    # Get AE col
                                    cols = df_scan[cup].columns
                                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                                    if not ae_col: continue
                                    
                                    series = day_data[cup][ae_col[0]]
                                    
                                    # Check 1: Day Burning (Consumption > threshold between Sunrise+1h and Sunset-1h to be safe/strict?)
                                    try:
                                        day_mask = (series.index > sunrise) & (series.index < sunset)
                                        day_kwh = series[day_mask].sum()
                                        
                                        if day_kwh > 1.0: # Tolerance 1 kWh
                                            anomalies.append({
                                                "Data": d,
                                                "CUPS": CUPS_MAPPING.get(cup, cup),
                                                "Tipus": "Encesa Diürna",
                                                "Valor": f"{day_kwh:.2f} kWh",
                                                "Raw_CUPS": cup  # For plotting laters
                                            })
                                            total_excess_kwh += day_kwh
                                    except Exception:
                                        pass # Skip if index issues
                                        
                                if i % 10 == 0:
                                    progress_bar.progress(i / total_steps)
                                    
                            progress_bar.progress(1.0)
                            st.session_state['anomalies_found'] = anomalies
                            st.session_state['anomalies_total_kwh'] = total_excess_kwh
                            st.session_state['scan_years_last_run'] = scan_years
                            
                            # Calculate Calculation Base: Total Lighting Consumption in Selected Years
                            lighting_total_scan, _ = get_aggregated_data(df_scan, lighting_selected)
                            st.session_state['scan_lighting_total_kwh'] = lighting_total_scan.sum()
                        
                # Display results if available
                if 'anomalies_found' in st.session_state and st.session_state['anomalies_found']:
                    anomalies_df = pd.DataFrame(st.session_state['anomalies_found'])
                    
                    # Metrics
                    total_detected = st.session_state.get('anomalies_total_kwh', 0)
                    total_baseline = st.session_state.get('scan_lighting_total_kwh', 1) # Avoid div0
                    pct_excess = (total_detected / total_baseline) * 100 if total_baseline > 0 else 0
                    
                    col_alert, col_total, col_pct = st.columns([2, 1, 1])
                    col_alert.error(f"⚠️ S'han detectat {len(anomalies_df)} incidències.")
                    col_total.metric("Energia Malbaratada (Est.)", f"{total_detected:.2f} kWh")
                    col_pct.metric("% sobre Total Enllumenat", f"{pct_excess:.2f} %")
                    
                    st.caption("ℹ️ Nota: Aquest càlcul és una estimació basada en dades horàries. La precisió exacta de l'encesa/apagada depèn de la resolució de les dades (horària vs minutal).")
                    
                    # Selector to view specific anomaly
                    # Sort by date desc
                    anomalies_df = anomalies_df.sort_values(by="Data", ascending=False)
                    anomaly_options = [f"{row['Data']} - {row['CUPS']} ({row['Valor']})" for index, row in anomalies_df.iterrows()]
                    selected_anomaly_str = st.selectbox("Seleccionar Incidència per visualitzar", anomaly_options)
                    
                    # Parse selection to get Date and CUPS
                    if selected_anomaly_str:
                        # Extract Date and CUPS safely
                        # String format: YYYY-MM-DD - CUPSNAME (X.XX kWh)
                        # Let's find the row corresponding to this string
                        idx = anomaly_options.index(selected_anomaly_str)
                        selected_row = anomalies_df.iloc[idx]
                        
                        view_date = selected_row['Data']
                        view_cups = selected_row['CUPS']
                else:
                    st.info("Prem el botó per buscar incidències.")
                    view_date = st.date_input("O selecciona una data manualment", min_csv_date + timedelta(days=1))
                    view_cups = st.selectbox("Seleccionar CUPS manualment", lighting_selected)

                # --- Visualizer (Shared logic) ---
                st.markdown("---")
                st.subheader(f"🔍 Detall: {view_cups} el {view_date}")
                
                # Setup Location (Girona)
                city = LocationInfo("Girona", "Catalonia", "Europe/Madrid", 41.9, 2.8)
                s = sun(city.observer, date=view_date)
                sunrise = s['sunrise'].replace(tzinfo=None)
                sunset = s['sunset'].replace(tzinfo=None)
                
                day_data = df.loc[df.index.date == view_date]
                
                if not day_data.empty:
                    cols = df[view_cups].columns
                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c][0]
                    day_series = day_data[view_cups][ae_col]
                    
                    fig_audit = go.Figure()
                    # Ensure index is naive to match sunrise/sunset
                    plot_index = day_series.index.tz_localize(None)
                    fig_audit.add_trace(go.Scatter(x=plot_index, y=day_series, name='Consum', fill='tozeroy', line=dict(color='#FF5733')))
                    
                    # Add background shading for Night
                    fig_audit.add_vrect(x0=plot_index[0], x1=sunrise, fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Nit")
                    fig_audit.add_vrect(x0=sunset, x1=plot_index[-1], fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Nit")
                    
                    # Force 1 hour ticks
                    fig_audit.update_layout(
                        title=f"Corba de Càrrega: {view_cups}",
                        xaxis=dict(
                            tickmode='linear',
                            dtick=3600000, # 1 hour
                            tickformat="%H:%M",
                            title="Hora",
                            range=[day_series.index[0], day_series.index[-1]] # Fix range to full day
                        ),
                        yaxis_title="kWh"
                    )
                    st.plotly_chart(fig_audit, use_container_width=True)
                else:
                    st.warning("No hi ha dades per aquesta data.")

        # --- Tab 4: AI Advisor ---
        with tab4:
            st.header("🤖 Assistent Virtual")
            
            with st.expander("ℹ️ Com funciona aquest assistent?"):
                st.markdown("""
                Aquest assistent aplica regles heurístiques per detectar patrons anòmals o interessants:
                
                **1. Encesa Diürna Sistemàtica (Enllumenat):**
                *   Busca quadres d'enllumenat que consumeixen >15% del total entre 10h i 16h (indicador de rellotge espatllat).
                
                **2. Consum Nocturn Elevat (Edificis):**
                *   Analitza edificis (escoles, pavellons...) que tenen un consum "base" nocturn (00h-06h) superior al 20% del dia. Pot indicar climatització encesa o equips fantasma.
                
                **3. Calendari Escolar i Festius (Escoles/Llar):**
                *   Revisa dies on l'escola hauria d'estar tancada (caps de setmana i festius comuns). Si el consum supera el 30% d'un dia laborable normal, s'alerta.
                
                **4. Anàlisi Estacional:**
                *   Compara l'hivern (Des-Feb) amb l'estiu (Jun-Ago) per determinar si l'edifici és "Hivernal" (Calefacció) o "Estival" (Aire Condicionat).
                """)
            
            # Initialize session state for advisor
            if 'show_advanced_advisor' not in st.session_state:
                st.session_state.show_advanced_advisor = False

            if st.button("Executar Anàlisi Avançada"):
                st.session_state.show_advanced_advisor = True
            
            if st.session_state.show_advanced_advisor:
                st.write("---")
                
                # --- YEAR FILTER ---
                all_years = sorted(df.index.year.unique().tolist())
                selected_years = st.multiselect("Filtrar per Anys", all_years, default=all_years)
                
                # Filter DF for Advisor
                if selected_years:
                    advisor_df = df[df.index.year.isin(selected_years)]
                else:
                    advisor_df = df
                    st.warning("Selecciona almenys un any.")

                with st.spinner("Analitzant patrons complexos..."):
                    
                    # --- 1. NIGHT CONSUMPTION CHECK (BUILDINGS) ---
                    st.subheader("🌙 Consum Nocturn Elevat (Edificis)")
                    night_anomalies = []
                    
                    # Use all dates
                    unique_dates = sorted(list(set(advisor_df.index.date)))
                    
                    # Pre-calculate building list
                    # Only check if CUPS is in building_cups and NOT simply a default name if possible, 
                    # but building_cups is robust enough based on classification.
                    
                    for d in unique_dates:
                         day_data = advisor_df.loc[advisor_df.index.date == d]
                         
                         for cup in building_cups: # Only buildings
                             cols = advisor_df[cup].columns
                             ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                             if not ae_col: continue
                             series = day_data[cup][ae_col[0]]
                             
                             if series.sum() > 5.0: # Minimum relevance threshold (5 kWh total day)
                                 night_mask = (series.index.hour >= 0) & (series.index.hour < 6)
                                 night_kwh = series[night_mask].sum()
                                 total_day = series.sum()
                                 
                                 if (night_kwh / total_day) > 0.20:
                                     night_anomalies.append({
                                         "Data": d, "CUPS": cup, "Nocturn": f"{night_kwh:.2f} kWh", 
                                         "Total": f"{total_day:.2f} kWh", "Rati": f"{(night_kwh/total_day*100):.1f}%"
                                     })
                    
                    if night_anomalies:
                        df_night = pd.DataFrame(night_anomalies)
                        st.warning(f"S'han detectat {len(df_night)} dies amb consum nocturn elevat (>20%).")
                        
                        # Interactive View
                        opt_night = [f"{r['Data']} - {r['CUPS']} ({r['Rati']})" for i, r in df_night.iterrows()]
                        sel_night = st.selectbox("Veure Detall Nocturn", opt_night)
                        
                        if sel_night:
                            idx_n = opt_night.index(sel_night)
                            row_n = df_night.iloc[idx_n]
                            viewing_date_n = row_n['Data']
                            viewing_cup_n = row_n['CUPS']
                            
                            # Plot logic (Quick Re-use)
                            dd_n = advisor_df.loc[advisor_df.index.date == viewing_date_n]
                            cc_n = advisor_df[viewing_cup_n].columns
                            ae_n = [c for c in cc_n if 'AE' in c and 'kWh' in c][0]
                            s_n = dd_n[viewing_cup_n][ae_n]
                            
                            fig_n = px.line(s_n, title=f"Consum Nocturn: {viewing_cup_n} ({viewing_date_n})", markers=True)
                            # Highlight night
                            fig_n.add_vrect(x0=s_n.index[0], x1=s_n.index[0].replace(hour=6), fillcolor="blue", opacity=0.1, annotation_text="0h-6h")
                            fig_n.update_layout(xaxis=dict(dtick=3600000, tickformat="%H:%M"))
                            st.plotly_chart(fig_n, use_container_width=True)
                    else:
                        st.success("✅ No s'han detectat consums nocturns anòmals en edificis.")

                    st.markdown("---")

                    # --- 2. SCHOOL CALENDAR CHECK ---
                    st.subheader("🎓 Anàlisi Calendari Escolar")
                    school_keywords = ['escola', 'llar', 'col·legi', 'institut']  # Keywords
                    
                    # Auto-detect
                    detected_schools = []
                    import re
                    for c in all_cups:
                         # c is the name now (because of parse_data change or fallback)
                         if any(k in c.lower() for k in school_keywords):
                            detected_schools.append(c)
                    
                    # Manual Selector Override
                    selected_schools = st.multiselect("Seleccionar Escoles/Llars manualment:", all_cups, default=detected_schools)
                    
                    school_anomalies = []
                    
                    if not selected_schools:
                        st.info("Selecciona els CUPS que corresponen a escoles per analitzar-los.")
                    else:
                        st.write(f"Analitzant: {', '.join(selected_schools)}")
                        
                        for c in selected_schools:
                            days_data = [] 
                            cols = advisor_df[c].columns
                            ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                            daily_sums = advisor_df[c][ae_col].resample('1d').sum() 
                            
                            # Baseline: Mean of Tues/Wed/Thu
                            weekdays_mask = (daily_sums.index.dayofweek.isin([1,2,3])) 
                            baseline = daily_sums[weekdays_mask].mean()
                            if pd.isna(baseline) or baseline == 0: baseline = 1.0 
                            
                            # Iterate days
                            for d_ts, kwh_val in daily_sums.items():
                                is_weekend = d_ts.dayofweek >= 5 
                                is_holiday = (d_ts.month == 8) or (d_ts.month == 12 and d_ts.day > 23)
                                
                                if (is_weekend or is_holiday):
                                    if kwh_val > (baseline * 0.30) and kwh_val > 5.0: 
                                        school_anomalies.append({
                                            "Data": d_ts.date(), 
                                            "CUPS": c,
                                            "Motiu": "Cap de Setmana/Festiu" if is_weekend else "Vacances",
                                            "Consum": f"{kwh_val:.2f} kWh",
                                            "Ref. Laborable": f"{baseline:.2f} kWh"
                                        })
                        
                        if school_anomalies:
                            df_school = pd.DataFrame(school_anomalies)
                            st.warning(f"S'han detectat {len(df_school)} dies amb consum alt en període inactiu.")
                            st.dataframe(df_school.sort_values("Data"), hide_index=True)
                            
                            # Interactive Viewer for School
                            opt_sch = [f"{r['Data']} - {r['CUPS']} ({r['Motiu']}) - {r['Consum']}" for i, r in df_school.iterrows()]
                            sel_sch = st.selectbox("Veure Gràfic Dia Escolar Anòmal", opt_sch)
                            
                            if sel_sch:
                                idx_s = opt_sch.index(sel_sch)
                                r_s = df_school.iloc[idx_s]
                                vd_s = r_s['Data']
                                target_cup_s = r_s['CUPS'] # Name or ID depending on logic
                                
                                # Plot
                                dd_s = advisor_df.loc[advisor_df.index.date == vd_s]
                                cc_s = advisor_df[target_cup_s].columns
                                ae_s_col = [x for x in cc_s if 'AE' in x and 'kWh' in x][0]
                                series_s = dd_s[target_cup_s][ae_s_col]
                                
                                fig_s = px.line(series_s, title=f"Consum Anòmal: {target_cup_s} ({vd_s}) - {r_s['Motiu']}", markers=True)
                                fig_s.update_layout(xaxis=dict(dtick=3600000, tickformat="%H:%M"))
                                st.plotly_chart(fig_s, use_container_width=True)
                        else:
                            st.success("✅ Les escoles semblen apagar correctament en festius.")

                    st.markdown("---")

                    # --- 3. SEASONAL PATTERN ANALYSIS ---
                    st.subheader("❄️☀️ Anàlisi Estacional (Hivern vs Estiu)")
                    
                    seasonal_data = []
                    for c in building_cups:
                        cols = advisor_df[c].columns
                        ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                        series = advisor_df[c][ae_col]
                        
                        # Winter: Dec, Jan, Feb
                        mask_winter = (series.index.month.isin([12, 1, 2]))
                        avg_winter = series[mask_winter].resample('1d').sum().mean()
                        
                        # Summer: Jun, Jul, Aug
                        mask_summer = (series.index.month.isin([6, 7, 8]))
                        avg_summer = series[mask_summer].resample('1d').sum().mean()
                        
                        # Avoid NaN
                        avg_winter = avg_winter if not pd.isna(avg_winter) else 0
                        avg_summer = avg_summer if not pd.isna(avg_summer) else 0
                        
                        classification = "Neutre"
                        if avg_winter > (avg_summer * 1.5):
                            classification = "🔴 Hivernal (Calefacció?)"
                        elif avg_summer > (avg_winter * 1.5):
                            classification = "🔵 Estival (Aire Cond.?)"
                            
                        seasonal_data.append({
                            "CUPS": c, # Use c as it might be name now
                            "Mitjana Hivern": f"{avg_winter:.1f} kWh",
                            "Mitjana Estiu": f"{avg_summer:.1f} kWh",
                            "Patró Detectat": classification
                        })
                        
                    st.table(pd.DataFrame(seasonal_data))

                    st.markdown("---")

                    # --- 4. TEMPORAL PATTERN ANALYSIS (WEEKLY & HOURLY) ---
                    st.subheader("📊 Anàlisi de Patrons Temporals (Setmanal i Horari)")
                    
                    temporal_data = []
                    
                    for c in building_cups:
                        cols = advisor_df[c].columns
                        ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                        series = advisor_df[c][ae_col]
                        
                        # 1. Weekly Pattern
                        # Resample by day first to get daily totals
                        daily = series.resample('1d').sum()
                        # Group by day of week (0=Mon, 6=Sun)
                        weekly_avg = daily.groupby(daily.index.dayofweek).mean()
                        
                        avg_workday = weekly_avg[0:5].mean()
                        avg_weekend = weekly_avg[5:7].mean()
                        
                        # Determine Pattern Type
                        if avg_workday > 0:
                            ratio_weekend = avg_weekend / avg_workday
                        else:
                            ratio_weekend = 1.0 # No consumption?
                            
                        patro_setmanal = "Indefinit"
                        if ratio_weekend < 0.4:
                            patro_setmanal = "🏢 Laborable (Tanca cap de setmana)"
                        elif ratio_weekend > 0.9:
                            patro_setmanal = "🔄 Continu (7 dies)"
                        else:
                            patro_setmanal = "Mixt"
                            
                        # 2. Hourly Pattern
                        # Group by hour
                        hourly_avg = series.groupby(series.index.hour).mean()
                        peak_hour = hourly_avg.idxmax()
                        
                        # Simple classification
                        if 8 <= peak_hour <= 18:
                            patro_horari = f"☀️ Diürn (Pic: {peak_hour}h)"
                        elif 19 <= peak_hour <= 23:
                            patro_horari = f"💡 Vespre (Pic: {peak_hour}h)"
                        else:
                            patro_horari = f"🌙 Nocturn (Pic: {peak_hour}h)"
                            
                        temporal_data.append({
                            "CUPS": c,
                            "Patró Setmanal": patro_setmanal,
                            "Patró Horari": patro_horari,
                            "Ratio CapDeSetmana": f"{ratio_weekend*100:.0f}%"
                        })
                        
                    st.table(pd.DataFrame(temporal_data))

        # === TAB 5: Autoconsum / Comunitat Energètica ===
        with tab5:
            st.header("☀️ Autoconsum i Comunitat Energètica")
            
            # Identify self-consumers from Whitelist directly
            # DF columns are Names, Whitelist is IDs. Reverse map needed.
            rev_map_local = {v: k for k, v in CUPS_MAPPING.items()}
            clean_whitelist = [x.strip().upper() for x in COMMUNITY_PARTICIPANTS]
            
            all_cols_idx = df.columns.get_level_values(0).unique()
            self_consumption_cups = []
            for c in all_cols_idx:
                cid = rev_map_local.get(c, c)
                if str(cid).strip().upper() in clean_whitelist:
                    self_consumption_cups.append(c)
            
            if not self_consumption_cups:
                st.info("No s'han detectat punts de la Comunitat Energètica amb dades d'autoconsum.")
            else:
                # Year Selection Filter (Tab 5 specific)
                available_years = sorted(df.index.year.unique())
                selected_years_t5 = st.multiselect("Filtrar per Anys", available_years, default=available_years, key='t5_year_filter')
                
                # Filter Data for Solar CUPS & Selected Years
                # Start with full DF (or filtered T1 if relevant, but T5 usually independent like Advisor)
                # Let's base on original DF to allow full year range selection independent of T1
                solar_df = df[self_consumption_cups]
                
                if selected_years_t5:
                     solar_df = solar_df[solar_df.index.year.isin(selected_years_t5)]
                else:
                     st.warning("Selecciona almenys un any.")
                     solar_df = pd.DataFrame() # Empty

                # Helpers for Self Consumption
                
                data_summary = []
                
                total_grid_kwh = 0
                total_self_kwh = 0
                
                for cup in self_consumption_cups:
                    cols = df[cup].columns
                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                    auto_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' in c]
                    
                    val_grid = 0
                    val_self = 0
                    
                    if ae_col: val_grid = solar_df[cup][ae_col[0]].sum() # Changed to solar_df to respect year filter
                    if auto_col: val_self = solar_df[cup][auto_col[0]].sum()
                    
                    # Logic Update: Grid IS Total.
                    total_demand = val_grid 
                    autarchy_pct = (val_self / total_demand * 100) if total_demand > 0 else 0
                    
                    net_grid = val_grid - val_self
                    
                    data_summary.append({
                        "CUPS": CUPS_MAPPING.get(cup, cup),
                        "Total Consum (kWh)": val_grid,
                        "Autoconsum (kWh)": val_self,
                        "Xarxa Facturat (kWh)": net_grid,
                        "% Autoconsum": autarchy_pct
                    })
                    
                    total_grid_kwh += val_grid
                    total_self_kwh += val_self
                    
                # Community Totals
                comm_demand = total_grid_kwh 
                comm_net_grid = comm_demand - total_self_kwh
                comm_autarchy = (total_self_kwh / comm_demand * 100) if comm_demand > 0 else 0
                
                # --- KPIs ---
                st.subheader("Balanç Global de la Comunitat (Total Participants)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Consum Edificis", f"{comm_demand:,.0f} kWh")
                k2.metric("Generació Autoconsumida", f"{total_self_kwh:,.0f} kWh")
                k3.metric("Xarxa (Facturat Estim.)", f"{comm_net_grid:,.0f} kWh")
                k4.metric("% Autoconsum Mitjà", f"{comm_autarchy:.1f}%")
                
                st.markdown("---")
                
                # --- Visualizations ---
                # Prepare time series
                ts_grid = pd.Series(0.0, index=solar_df.index)
                ts_self = pd.Series(0.0, index=solar_df.index)
                
                for c in self_consumption_cups:
                     cols = solar_df[c].columns 
                     ae_col = [x for x in cols if 'AE' in x and 'kWh' in x and 'AUTOCONS' not in x]
                     auto_col = [x for x in cols if 'AE' in x and 'kWh' in x and 'AUTOCONS' in x]
                     
                     if ae_col: ts_grid = ts_grid.add(solar_df[c][ae_col[0]], fill_value=0)
                     if auto_col: ts_self = ts_self.add(solar_df[c][auto_col[0]], fill_value=0)
                
                # Resample for Chart
                days = (solar_df.index.max() - solar_df.index.min()).days
                resample_freq = '1d' if days < 60 else '1W' if days < 365 else 'ME'
                
                chart_grid = ts_grid.resample(resample_freq).sum()
                chart_self = ts_self.resample(resample_freq).sum()
                
                # Force Full Year X-Axis if Single Year Selected
                if len(selected_years_t5) == 1:
                    target_y = selected_years_t5[0]
                    if resample_freq == 'ME':
                         full_idx = pd.date_range(start=f'{target_y}-01-01', end=f'{target_y}-12-31', freq='ME')
                         # ME might give end of month. If data is Monthly, this is fine.
                         # If data is misaligned, careful. Assuming ME aligns.
                         chart_grid = chart_grid.reindex(full_idx, fill_value=0)
                         chart_self = chart_self.reindex(full_idx, fill_value=0)
                         
                         # Format index nicely name?
                         # Plotly handles dates generally well.

                # Calculate Net Grid (Facturat) for Chart STACKING
                # Logic: Grid Column IS Total.
                # Stack: Self + Net = Total.
                chart_net_grid = chart_grid - chart_self
                chart_net_grid = chart_net_grid.clip(lower=0)
                
                # Line Chart Metric: % Autoconsum = Self / Total * 100
                chart_pct = (chart_self / chart_grid * 100).fillna(0)
                
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("##### ⚡ Evolució Mix Energètic (Comunitat)")
                    fig_bal = go.Figure()
                    fig_bal.add_trace(go.Bar(x=chart_self.index, y=chart_self, name='Autoconsum', marker_color='gold'))
                    # Stacked on top: Net Grid
                    fig_bal.add_trace(go.Bar(x=chart_net_grid.index, y=chart_net_grid, name='Consum equipament', marker_color='gray'))
                    
                    fig_bal.update_layout(barmode='stack', xaxis_title="Temps", yaxis_title="kWh", legend=dict(orientation="h", y=1.1))
                    # Ensure x-axis shows all months if single year (by explicit tickformat or just data presence)
                    if len(selected_years_t5) == 1 and resample_freq == 'ME':
                         fig_bal.update_xaxes(dtick="M1", tickformat="%b")
                    
                    st.plotly_chart(fig_bal, use_container_width=True)
                    
                    st.markdown("##### 📈 Evolució % Autoconsum")
                    fig_aut = px.line(x=chart_pct.index, y=chart_pct.values, markers=True)
                    fig_aut.update_traces(line_color='green')
                    fig_aut.update_layout(xaxis_title="Temps", yaxis_title="% Autoconsum")
                    if len(selected_years_t5) == 1 and resample_freq == 'ME':
                         fig_aut.update_xaxes(dtick="M1", tickformat="%b")
                    st.plotly_chart(fig_aut, use_container_width=True)
                    
                with c2:
                    st.markdown("##### 🍰 Quota Autoconsum per Participant")
                    df_summ = pd.DataFrame(data_summary)
                    if not df_summ.empty:
                        fig_pie = px.pie(df_summ, values='Autoconsum (kWh)', names='CUPS', hole=0.4)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("Sense dades.")

                st.markdown("---")
                st.subheader("Detall per Participant")
                st.dataframe(
                    df_summ.style.format({
                        "Total Consum (kWh)": "{:,.0f}", 
                        "Autoconsum (kWh)": "{:,.0f}", 
                        "Xarxa Facturat (kWh)": "{:,.0f}",
                        "% Autoconsum": "{:.1f}%"
                    })
                )

        # === TAB 6: FV Sala Nova (Exclusive) ===
        with tab6:
            st.header("☀️ Producció Fotovoltaica - Sala Nova (Base de Dades)")
            
            try:
                # 1. Load Data
                supa_client = init_supabase()
                # Use Global Cached Function (Optimized)
                df_fv = load_fv_sala_nova_data()
                
                if not df_fv.empty:
                    
                    # 2. Controls and State Management
                    if 't6_anchor_date' not in st.session_state:
                         st.session_state['t6_anchor_date'] = datetime.date.today()

                    c_ctrl1, c_nav1, c_nav2, c_nav3 = st.columns([2, 1, 3, 1])
                    with c_ctrl1:
                        # Time Scale Selector
                        time_scale = st.selectbox(
                            "Escala Temporal", 
                            ["Anual", "Mensual", "Setmanal", "Diària"], 
                            index=0,
                            key='t6_scale_selector'
                        )
                    
                    # Logic: Determine Shift Delta based on Scale
                    from dateutil.relativedelta import relativedelta

                    shift_delta = relativedelta(years=1)
                    if time_scale == "Mensual": shift_delta = relativedelta(months=1)
                    elif time_scale == "Setmanal": shift_delta = relativedelta(weeks=1)
                    elif time_scale == "Diària": shift_delta = relativedelta(days=1)

                    with c_nav1:
                        st.write("") # Spacer
                        if st.button("⬅️ Enrere", key="t6_prev"):
                            st.session_state['t6_anchor_date'] -= shift_delta
                            
                    with c_nav3:
                        st.write("") # Spacer
                        if st.button("Endavant ➡️", key="t6_next"):
                            st.session_state['t6_anchor_date'] += shift_delta
                            
                    # Calculate Date Range for Filtering
                    anchor = st.session_state['t6_anchor_date']
                    start_view = anchor
                    end_view = anchor 
                    title_view = str(anchor)
                    resample_freq = 'ME'

                    if time_scale == "Anual":
                         start_view = datetime.date(anchor.year, 1, 1)
                         end_view = datetime.date(anchor.year, 12, 31)
                         resample_freq = 'ME'
                         title_view = f"Any {anchor.year}"
                         
                    elif time_scale == "Mensual":
                         start_view = datetime.date(anchor.year, anchor.month, 1)
                         next_m = start_view + relativedelta(months=1)
                         end_view = next_m - timedelta(days=1)
                         resample_freq = 'd'
                         title_view = f"{month_names.get(anchor.month, anchor.month)} {anchor.year}"
                         
                    elif time_scale == "Setmanal":
                         start_view = anchor - timedelta(days=anchor.weekday())
                         end_view = start_view + timedelta(days=6)
                         resample_freq = 'd'
                         title_view = f"Setmana {start_view.strftime('%d/%m')} - {end_view.strftime('%d/%m/%Y')}"
                         
                    elif time_scale == "Diària":
                         start_view = anchor
                         end_view = anchor
                         resample_freq = 'h'
                         title_view = f"{anchor.strftime('%d/%m/%Y')}"

                    with c_nav2:
                         st.markdown(f"<h3 style='text-align: center; margin-top: 5px;'>{title_view}</h3>", unsafe_allow_html=True)

                    # Filter PV Data
                    mask_fv = (df_fv.index.date >= start_view) & (df_fv.index.date <= end_view)
                    df_fv_filtered = df_fv[mask_fv]
                    
                    # Resample PV Data
                    s_fv_resampled = df_fv_filtered['potencia_fv'].resample(resample_freq).sum() if not df_fv_filtered.empty else pd.Series()
                    if s_fv_resampled.empty:
                         # Create specific empty index for plot structure
                         # This avoids "No data" generic error if possible, shows empty chart
                         pass
                    
                    # 3. Chart 1: General PV Production
                    st.subheader(f"Generació Fotovoltaica ({time_scale})")
                    
                    fig_gen = go.Figure()
                    
                    # Determine Chart Type based on Scale
                    # Hourly/Daily usually Bar? Evolution usually Bar if discrete periods.
                    # Solar curve (Daily) nice as Area/Line.
                    chart_type = 'bar'
                    if time_scale == "Diària": chart_type = 'area' # Solar curve style
                    
                    if not s_fv_resampled.empty:
                        if chart_type == 'area':
                             fig_gen.add_trace(go.Scatter(
                                x=s_fv_resampled.index, 
                                y=s_fv_resampled, 
                                name="Producció FV",
                                fill='tozeroy',
                                line=dict(color='#FFC300')
                             ))
                        else:
                             fig_gen.add_trace(go.Bar(
                                x=s_fv_resampled.index, 
                                y=s_fv_resampled, 
                                name="Producció FV",
                                marker_color='#FFC300'
                             ))
                    else:
                        st.warning(f"No hi ha dades per al període: {title_view}")
                    
                    fig_gen.update_layout(
                        title=f"Producció FV - {title_view}",
                        xaxis_title="Temps",
                        yaxis_title="kWh",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_gen, use_container_width=True)
                    
                    # 4. Chart 2: Comparativa vs Community Demand
                    st.markdown("---")
                    st.subheader("Comparativa: Producció vs Demanda Comunitat")
                    
                    # Check if 'df' (Consumption) is available
                    if df is not None and not df.empty:
                        # Identify Community CUPS
                        rev_map_local = {v: k for k, v in CUPS_MAPPING.items()}
                        clean_whitelist = [x.strip().upper() for x in COMMUNITY_PARTICIPANTS]
                        
                        community_consumption_cups = []
                        all_cols_idx = df.columns.get_level_values(0).unique()
                        for c in all_cols_idx:
                            cid = rev_map_local.get(c, c)
                            if str(cid).strip().upper() in clean_whitelist:
                                community_consumption_cups.append(c)
                        
                        if community_consumption_cups:
                            # Filter and Aggregate Consumption
                            mask_df = (df.index.date >= start_view) & (df.index.date <= end_view)
                            df_filtered_view = df.loc[mask_df]
                            
                            ts_comm_demand = pd.Series(0.0, index=df_filtered_view.index)
                            
                            if not df_filtered_view.empty:
                                for c in community_consumption_cups:
                                    cols = df_filtered_view[c].columns
                                    # AE = Grid Consumption. Total = Grid + Self-Consumed.
                                    
                                    ae_col = [x for x in cols if 'AE' in x and 'kWh' in x and 'AUTOCONS' not in x]
                                    auto_col = [x for x in cols if 'AE' in x and 'kWh' in x and 'AUTOCONS' in x]
                                    
                                    if ae_col: ts_comm_demand = ts_comm_demand.add(df_filtered_view[c][ae_col[0]], fill_value=0)
                                    if auto_col: ts_comm_demand = ts_comm_demand.add(df_filtered_view[c][auto_col[0]], fill_value=0)
                            
                            # Resample Consumption
                            # Use correct filtered DF freq or new freq?
                            # freq is global for this tab from 'time_scale'
                            # resample_freq was calculated earlier
                            
                            s_comm_resampled = ts_comm_demand.resample(resample_freq).sum() if not ts_comm_demand.empty else pd.Series()
                            
                            fig_comp_comm = go.Figure()
                            
                            # Trace 1: Demand (Area/Line)
                            if not s_comm_resampled.empty:
                                fig_comp_comm.add_trace(go.Scatter(
                                    x=s_comm_resampled.index,
                                    y=s_comm_resampled,
                                    name="Demanda Comunitat",
                                    mode='lines',
                                    fill='tozeroy',
                                    line=dict(color='gray', width=1),
                                    fillcolor='rgba(128, 128, 128, 0.2)'
                                ))

                            
                            # Trace 2: Generation (Line/Bar)
                            fig_comp_comm.add_trace(go.Scatter(
                                x=s_fv_resampled.index,
                                y=s_fv_resampled,
                                name="Producció FV Sala Nova",
                                line=dict(color='#FFC300', width=2)
                            ))
                            
                            fig_comp_comm.update_layout(
                                title=f"Cobertura de Demanda ({time_scale})",
                                xaxis_title="Temps",
                                yaxis_title="kWh",
                                hovermode="x unified",
                                legend=dict(orientation="h", y=1.1)
                            )
                            st.plotly_chart(fig_comp_comm, use_container_width=True)
                            
                        else:
                             st.warning("No s'han trobat CUPS de la comunitat a les dades de consum carregades.")
                    else:
                        st.info("Carrega dades de consum (CSV/DB) per veure la comparativa de demanda.")
                    
                else:
                    st.info("Encara no hi ha dades a la taula FV_Sala_Nova.")
            except Exception as e:
                st.error(f"Error carregant dades FV: {e}")

        # === TAB 7: Optimitzador CLE ===
        with tab7:
            render_cle_optimizer()


if __name__ == '__main__':
    main()
