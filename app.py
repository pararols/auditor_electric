import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from astral import LocationInfo
from astral.sun import sun
import datetime
from datetime import timedelta
import numpy as np

# Page Config
st.set_page_config(
    page_title="Auditor Energètic & Enllumenat Públic",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
CUPS_MAPPING = {
    "ES0031408137509001NN0F": "Deixalleria",
    "ES0031406053348001SC0F": "Enll C/ Llevant-Estació",
    "ES0031406053355001KM0F": "Enll Veïnat Nou",
    "ES0031406053357001QG0F": "Sala Nova",
    "ES0031406053359002BG0F": "Enll Centre Poble",
    "ES0031406053362001AJ0F": "Enll Mas Masó",
    "ES0031406053560001XY0F": "Escola",
    "ES0031406054170001JT0F": "Ajuntament",
    "ES0031406054364001YH0F": "Enll Sobrànigues",
    "ES0031406056222001JD0F": "Enll Estació",
    "ES0031406056223001XC0F": "Enll C/ generalitat",
    "ES0031406115758001TA0F": "Enll- Diana",
    "ES0031406233593001BT0F": "Correus",
    "ES0031406267955002TR0F": "Camp futbol i vesturaris",
    "ES0031408030887001SD0F": "Enll-Bon repòs",
    "ES0031408303814001QQ0F": "Llar Infants",
    "ES0031408305363001CN0F": "Enll Rotonda crta",
    "ES0031408332025001ZK0F": "Polivalent",
    "ES0031408457126001XL0F": "Pavelló",
    "ES0031408528667001SW0F": "Enll C/ nou",
    "ES0031408691405001KF0F": "Can Burcet",
}

# --- CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

def parse_data(uploaded_file):
    """
    Parses the specific Multi-Index Header CSV format manually to avoid header length mismatches.
    Row 0: CUPS IDs (Merged cells need forward fill)
    Row 1: Variable names (AE_kWh, AE_AUTOCONS_kWh)
    """
    try:
        uploaded_file.seek(0)
        # Read as string/object first to handle headers safely
        # engine='python' is more robust for separators at end of lines etc.
        df_raw = pd.read_csv(uploaded_file, header=None, sep=';', dtype=object, engine='python')
        
        # 1. Extract Headers
        header_r0 = df_raw.iloc[0].copy()
        header_r1 = df_raw.iloc[1].copy()
        
        # Forward fill CUPS in row 0
        header_r0 = header_r0.ffill()
        
        # 2. Extract Data
        df_data = df_raw.iloc[2:].copy()
        
        # 3. Construct MultiIndex Columns
        new_columns = []
        for i in range(len(header_r0)):
            h0 = header_r0.iloc[i]
            h1 = header_r1.iloc[i]
            
            # Handle standard columns (Fecha/Hora) which might have empty H0
            if pd.isna(h0) or str(h0).strip() == '':
                h0 = 'Metadata'
            else:
                h0 = str(h0).strip()
                
            # Apply Mapping for CUPS
            if h0 != 'Metadata':
                # Use mapping if exists, otherwise keep original ID (Don't merge all into CAN BURCET)
                h0 = CUPS_MAPPING.get(h0, h0)
            
            # Clean stripping level 1 and ensure string
            if pd.isna(h1):
                h1 = ""
            else:
                h1 = str(h1).strip()
                
            new_columns.append((h0, h1))
            
        df_data.columns = pd.MultiIndex.from_tuples(new_columns)
        
        # 4. Process Date and Time
        # Identify columns
        # We look for 'Fecha' and 'Hora' in level 1, or just take first two if they look like it
        # Based on image: Col 0 is Fecha, Col 1 is Hora.
        
        # Helper to find column by level 1 name (case insensitive)
        def get_col_by_l1(name, columns):
            for c in columns:
                if isinstance(c[1], str) and name.lower() in c[1].lower():
                    return c
            return None

        date_col = get_col_by_l1('Fecha', df_data.columns)
        time_col = get_col_by_l1('Hora', df_data.columns)
        
        if not date_col or not time_col:
            # Fallback to pos 0 and 1
            date_col = df_data.columns[0]
            time_col = df_data.columns[1]

        # Combine
        # Standardize separators just in case
        dates = df_data[date_col].astype(str)
        times = df_data[time_col].astype(str)
        
        # Handle '24' hour or other quirks if needed, but assuming standard 0-23
        # In Spain sometimes 1-24 is used. Image shows '0', '1', '2'... so standard 0-23 or 1-24?
        # Image shows '0'.
        
        # If time is just an integer hour (0, 1, 2), convert to HH:00
        # If it's already HH:MM, fine.
        # Let's clean the time string.
        def clean_time(t):
            t = t.strip()
            if t.isdigit(): # "0", "1"
                return f"{int(t):02d}:00"
            return t # Assume format is okay
            
        times = times.apply(clean_time)
        
        datetime_str = dates + ' ' + times
        
        # Create Index directly
        datetime_index = pd.to_datetime(datetime_str, dayfirst=True, errors='coerce')
        
        # Assign index
        df_data.index = datetime_index
        df_data.index.name = 'Datetime'
        
        # Drop rows with invalid dates (NaT)
        df_data = df_data[df_data.index.notna()]
        
        # Drop the original metadata columns
        df_data = df_data.drop(columns=[date_col, time_col])
        
        # 5. Convert Numeric Columns
        # Replace decimal ',' with '.' and cast
        for col in df_data.columns:
            # col is a tuple (CUPS, Var)
            if df_data[col].dtype == object:
                # Replace comma with dot
                df_data[col] = df_data[col].astype(str).str.replace('.', '', regex=False) # Remove thousand separators if any?
                # Wait, European: 1.000,00 -> remove dot, replace comma
                # But CSV usually simple. User said: ", com a decimal".
                # If there are thousands separators (.), we should remove them.
                # But careful not to remove dot if it's not there.
                # Assuming simple format "0,065".
                
                # Safer: replace ',' with '.'
                val_series = df_data[col].astype(str).str.replace(',', '.', regex=False)
                df_data[col] = pd.to_numeric(val_series, errors='coerce').fillna(0)
                
        return df_data

    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def classify_cups_by_name(df):
    """
    Classifies CUPS into 'Building' or 'Public Lighting' based on Name.
    Public Lighting: Starts with 'Enll' (case insensitive).
    """
    cups_list = df.columns.get_level_values(0).unique()
    
    lighting = []
    buildings = []
    
    for cups in cups_list:
        # Check if name starts with 'Enll'
        if str(cups).lower().startswith("enll"):
            lighting.append(cups)
        else:
            buildings.append(cups)
            
    return lighting, buildings

# --- New Helper: Date Navigator ---
def get_date_range(view_mode, anchor_date):
    """Returns (start_date, end_date, freq_alias) based on view mode and anchor."""
    start_date = None
    end_date = None
    freq = 'h' # default
    
    if view_mode == 'Diària':
        start_date = anchor_date
        end_date = anchor_date
        freq = '1h'
    elif view_mode == 'Setmanal':
        start_date = anchor_date - timedelta(days=anchor_date.weekday())
        end_date = start_date + timedelta(days=6)
        freq = '1d' # Daily bars for weekly view
    elif view_mode == 'Mensual':
        start_date = anchor_date.replace(day=1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        freq = '1d'
    elif view_mode == 'Anual':
        start_date = anchor_date.replace(month=1, day=1)
        end_date = anchor_date.replace(month=12, day=31)
        freq = 'ME' # Monthly bars
        
    return start_date, end_date, freq

def shift_date(view_mode, anchor_date, direction):
    """Shifts the anchor date forward or backward."""
    if view_mode == 'Diària':
        return anchor_date + timedelta(days=direction)
    elif view_mode == 'Setmanal':
        return anchor_date + timedelta(weeks=direction)
    elif view_mode == 'Mensual':
        # Shift month
        new_month = anchor_date.month + direction
        year_adj = 0
        if new_month > 12:
            new_month = 1
            year_adj = 1
        elif new_month < 1:
            new_month = 12
            year_adj = -1
        return anchor_date.replace(year=anchor_date.year + year_adj, month=new_month, day=1)
    elif view_mode == 'Anual':
        return anchor_date.replace(year=anchor_date.year + direction)
    return anchor_date

# --- Main App Interface ---

def main():
    st.title("💡 Auditor Energètic & Enllumenat Públic")
    
    # Init Session State for Filters
    if 'selected_cups_list' not in st.session_state:
        st.session_state.selected_cups_list = []
    if 'anchor_date' not in st.session_state:
        st.session_state.anchor_date = datetime.date.today()
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'Mensual'

    # Sidebar
    st.sidebar.header("Pujar Dades")
    uploaded_file = st.sidebar.file_uploader("Pujar CSV (Format Horari)", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner('Processant dades...'):
            df = parse_data(uploaded_file)
            
        if df is not None:
            st.success("Dades carregades correctament!")
            
            # Use data range to initialize anchor if outside range
            min_csv_date = df.index.min().date()
            max_csv_date = df.index.max().date()
            
            # Simple check to init anchor date inside range if it's currently outside
            if not (min_csv_date <= st.session_state.anchor_date <= max_csv_date):
                 st.session_state.anchor_date = min_csv_date

            # --- Classification Step (Name Based) ---
            st.subheader("🤖 Classificació Automàtica de CUPS")
            lighting_cups, building_cups = classify_cups_by_name(df)
            all_cups = df.columns.get_level_values(0).unique().tolist()
            
            # Show Classification Logic Results
            with st.expander("Veure Detall Classificació (Enllumenat vs Edificis)", expanded=True):
                col_class_1, col_class_2 = st.columns(2)
                
                # Reverse Map for Display
                rev_map = {v: k for k, v in CUPS_MAPPING.items()}
                
                def make_display_df(items):
                    rows = []
                    for it in items:
                        # it is the name in the DF (Mapped or Raw)
                        real_cups = rev_map.get(it, it) # Get key if it was a value, else it
                        # If mapped, it is key -> value. So if 'it' is 'Can Burcet', rev is CUPS. 
                        # If 'it' is CUPS (notmapped), rev_map.get(it) is None usually unless identity.
                        # Wait, CUPS_MAPPING is ID -> Name. 
                        # 'it' is Name (if mapped) or ID (if not).
                        # If 'it' is Name, we want ID.
                        
                        # Check if 'it' is in values of map
                        found_key = None
                        for k, v in CUPS_MAPPING.items():
                            if v == it:
                                found_key = k
                                break
                        
                        if found_key:
                             display_cups = found_key
                        else:
                             display_cups = it # Likely the ID itself
                             
                        rows.append({"Nom Assignat": it, "CUPS Original": display_cups})
                    return pd.DataFrame(rows)
                
                with col_class_1:
                    st.markdown(f"**💡 Enllumenat ({len(lighting_cups)})**")
                    st.dataframe(make_display_df(lighting_cups), hide_index=True, use_container_width=True)
                    
                with col_class_2:
                    st.markdown(f"**🏢 Edificis / Altres ({len(building_cups)})**")
                    st.dataframe(make_display_df(building_cups), hide_index=True, use_container_width=True)
            
            # --- Global Filter Logic Helpers ---
            def set_cups_selection(group_type):
                if group_type == 'All':
                    st.session_state.selected_cups_list = all_cups
                elif group_type == 'Lighting':
                    st.session_state.selected_cups_list = lighting_cups
                elif group_type == 'Buildings':
                    st.session_state.selected_cups_list = building_cups
                else:
                    st.session_state.selected_cups_list = []

            # Initialize selection if empty on first load
            if not st.session_state.selected_cups_list:
                 st.session_state.selected_cups_list = all_cups

            # --- Tabs ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Panell Global", "📈 Comparativa", "🌃 Auditor Enllumenat", "🤖 AI Advisor"])
            
            # === TAB 1: Global Dashboard ===
            with tab1:
                st.header("Visió General")
                
                # 1. Controls Row
                col_nav1, col_nav2, col_nav3 = st.columns([2, 1, 3])
                
                with col_nav1:
                     # View Mode
                     mode = st.selectbox("Escala Temporal", ["Diària", "Setmanal", "Mensual", "Anual"], key="view_mode_t1")
                
                with col_nav2:
                    # Navigation Buttons
                    col_b1, col_b2 = st.columns(2)
                    if col_b1.button("⬅️", key="prev_t1"):
                        st.session_state.anchor_date = shift_date(mode, st.session_state.anchor_date, -1)
                    if col_b2.button("➡️", key="next_t1"):
                        st.session_state.anchor_date = shift_date(mode, st.session_state.anchor_date, 1)

                # Calculate Date Range
                start_d, end_d, freq_alias = get_date_range(mode, st.session_state.anchor_date)
                
                with col_nav3:
                    st.subheader(f"📅 {start_d} - {end_d}")

                # 2. Filter Data based on time
                # Filter rows
                mask_time = (df.index.date >= start_d) & (df.index.date <= end_d)
                df_filtered_t1 = df.loc[mask_time]
                
                # Filter columns (Using 'All' implicitly for Global view, or let user filter?)
                # Global view usually implies everything, but user might want to see specific aggregate
                # Let's add basic CUPS filter here too or just use ALL?
                # User asked for "Visio General" simply. Let's aggregate ALL selected cups from session state if relevant,
                # but usually global implies Site Total. Let's use the Session State list to allow flexibility.
                
                # --- Quick Selectors (re-used logic visually, but maybe less prominent here?) 
                # User asked specifically for these buttons in Comparative, but implied flexibility in Global.
                # Let's put a Multiselect here for Global Aggregation.
                
                selected_cups_t1 = st.multiselect("Filtrar CUPS (Agregat)", all_cups, default=all_cups, key="sel_cups_t1")
                
                # Aggregation
                # Helper to get AE and Self-Consumption for selected CUPS
                def get_aggregated_data(data_df, cups_list):
                    total_ae = pd.Series(0, index=data_df.index)
                    total_autocons = pd.Series(0, index=data_df.index)
                    for cups in cups_list:
                        cols = data_df[cups].columns
                        ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                        autocons_col = [c for c in cols if 'AUTOCONS' in c]
                        if ae_col: total_ae = total_ae.add(data_df[cups][ae_col[0]], fill_value=0)
                        if autocons_col: total_autocons = total_autocons.add(data_df[cups][autocons_col[0]], fill_value=0)
                    return total_ae, total_autocons

                agg_ae, agg_autocons = get_aggregated_data(df_filtered_t1, selected_cups_t1)
                
                # 3. Display KPIs & Chart
                if not agg_ae.empty:
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Consum Xarxa", f"{agg_ae.sum():,.2f} kWh")
                    kpi2.metric("Autoconsum", f"{agg_autocons.sum():,.2f} kWh")
                    total_demand = agg_ae.sum() + agg_autocons.sum()
                    kpi3.metric("Autosuficiència", f"{(agg_autocons.sum()/total_demand*100 if total_demand > 0 else 0):.2f} %")
                    
                    # Resample for Chart
                    # If freq_alias == '1h' (Daily view), keep as is.
                    # If '1d' (Weekly/Monthly), resample sum.
                    # If 'ME' (Yearly), resample sum.
                    
                    chart_ae = agg_ae
                    chart_ac = agg_autocons
                    
                    if freq_alias != 'h': # Not raw
                        chart_ae = agg_ae.resample(freq_alias).sum()
                        chart_ac = agg_autocons.resample(freq_alias).sum()
                        
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Bar(x=chart_ae.index, y=chart_ae, name='Consum Xarxa', marker_color='#EF553B') if freq_alias != '1h' else go.Scatter(x=chart_ae.index, y=chart_ae, name='Consum Xarxa', line=dict(color='#EF553B'), fill='tozeroy'))
                    
                    if agg_autocons.sum() > 0:
                         fig_line.add_trace(go.Bar(x=chart_ac.index, y=chart_ac, name='Autoconsum', marker_color='#00CC96') if freq_alias != '1h' else go.Scatter(x=chart_ac.index, y=chart_ac, name='Autoconsum', line=dict(color='#00CC96'), fill='tozeroy'))

                    # Determine dtick for x-axis based on view mode
                    xaxis_args = {'title': "Temps"}
                    if mode == 'Diària':
                        xaxis_args['dtick'] = 3600000 * 1 # 1 hour in ms
                        xaxis_args['tickformat'] = "%H:%M"
                    elif mode == 'Setmanal':
                        xaxis_args['dtick'] = 86400000.0 # 1 day
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

            # === TAB 2: Comparative Analysis ===
            with tab2:
                st.header("Anàlisi Comparatiu")
                
                # 1. Quick Selectors
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                
                # Helper to update the multiselect state
                def update_selection(new_list):
                    st.session_state["multi_comp"] = new_list
                
                if col_btn1.button("Tots (All)", key="btn_all"): 
                    update_selection(all_cups)
                if col_btn2.button("Enllumenat", key="btn_light"): 
                    update_selection(lighting_cups)
                if col_btn3.button("Edificis (Resta)", key="btn_build"): 
                    update_selection(building_cups)
                if col_btn4.button("Netejar", key="btn_clear"): 
                    update_selection([])
                
                # Ensure the key exists before multiselect if not already (safeguard)
                if "multi_comp" not in st.session_state:
                     st.session_state["multi_comp"] = all_cups
                
                # Multiselect sync with buttons
                current_selection = st.multiselect("CUPS Seleccionats", all_cups, key="multi_comp")
                
                # 2. Controls (Time) - Independent from Tab 1? 
                # It's better to share Anchor Date usually, but maybe distinctive Modes.
                # Let's reuse the same anchor date principle for consistency.
                st.markdown("---")
                col_c1, col_c2, col_c3 = st.columns([2, 1, 3])
                with col_c1:
                    mode_t2 = st.selectbox("Escala Temporal (Comparativa)", ["Diària", "Setmanal", "Mensual", "Anual"], key="mode_t2")
                with col_c2:
                    col_cb1, col_cb2 = st.columns(2)
                    if col_cb1.button("⬅️", key="prev_t2"):
                        st.session_state.anchor_date = shift_date(mode_t2, st.session_state.anchor_date, -1)
                    if col_cb2.button("➡️", key="next_t2"):
                         st.session_state.anchor_date = shift_date(mode_t2, st.session_state.anchor_date, 1)
                
                start_d2, end_d2, freq_alias2 = get_date_range(mode_t2, st.session_state.anchor_date)
                 
                with col_c3:
                     st.subheader(f"📅 {start_d2} - {end_d2}")
                     
                # 3. Bar Chart (Stacked by CUPS)
                # Filter time
                mask_time2 = (df.index.date >= start_d2) & (df.index.date <= end_d2)
                df_filtered_t2 = df.loc[mask_time2]
                
                if not df_filtered_t2.empty and current_selection:
                    # Prepare Data for Plotly Express
                    # We need a long-format DF: [Datetime, CUPS, kWh]
                    # Loop selected cups
                    plot_data = []
                    
                    for cup in current_selection:
                        # Get AE col
                        cols = df_filtered_t2[cup].columns
                        ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                        
                        if ae_col:
                            series = df_filtered_t2[cup][ae_col[0]]
                            
                            # Resample if needed
                            if freq_alias2 != 'h':
                                series = series.resample(freq_alias2).sum()
                            
                            # Create small DF
                            tmp = series.reset_index()
                            tmp.columns = ['Datetime', 'kWh']
                            tmp['CUPS'] = cup
                            plot_data.append(tmp)
                    
                    if plot_data:
                        final_plot_df = pd.concat(plot_data)
                        
                        fig_bar = px.bar(
                            final_plot_df, 
                            x='Datetime', 
                            y='kWh', 
                            color='CUPS', 
                            title=f"Consum Desglossat ({mode_t2})",
                            text_auto='.2s' if len(current_selection) < 5 else False
                        )
                        
                        # Axis settings for bar chart
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
                    
                # 4. Flexible Comparison Chart (Series)
                st.subheader("Comparativa de Sèries (Tendències)")
                
                # Custom container for controls
                col_sel, col_res = st.columns([4, 1])
                
                # Define Options
                comp_options = ["TOTAL", "ENLLUMENAT (Agregat)", "EDIFICIS (Agregat)"] + sorted(all_cups)
                
                # Session state for this specific selector
                if "comp_series_sel" not in st.session_state:
                    st.session_state.comp_series_sel = ["TOTAL"]
                    
                # Reset Button logic
                if col_res.button("Restablir (Total)", key="btn_reset_comp"):
                    st.session_state.comp_series_sel = ["TOTAL"]
                
                # Multiselect
                selection_series = col_sel.multiselect("Afegir/Treure Sèries al Gràfic", comp_options, key="series_multiselect", default=st.session_state.comp_series_sel)
                
                # Year Selector for Comparison
                available_years = sorted(df.index.year.unique())
                col_y_sel, _ = st.columns([2, 3])
                selected_years_comp = col_y_sel.multiselect("Seleccionar Anys a Comparar", available_years, default=available_years, key="years_comp_sel")

                # Plotting Logic
                if selection_series:
                    fig_comp = go.Figure()
                    
                    # Logic: We want to compare across years based on the current View Mode.
                    current_anchor = st.session_state.anchor_date
                    
                    if not selected_years_comp:
                        st.warning("Selecciona almenys un any per comparar.")
                    else:
                        for item in selection_series:
                            # 1. Get Series Data (Full History)
                            if item == "TOTAL":
                                s_full, _ = get_aggregated_data(df, all_cups)
                            elif item == "ENLLUMENAT (Agregat)":
                                s_full, _ = get_aggregated_data(df, lighting_cups)
                            elif item == "EDIFICIS (Agregat)":
                                s_full, _ = get_aggregated_data(df, building_cups)
                            else:
                                if item in all_cups:
                                    cols = df[item].columns
                                    ae_col = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                                    if ae_col:
                                        s_full = df[item][ae_col[0]]
                                    else:
                                        continue
                            
                            # 2. Slice and Plot based on Mode
                            if mode_t2 == 'Anual':
                                # Resample to Monthly first
                                s_monthly = s_full.resample('ME').sum()
                                
                                month_map = {1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
                                             7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Oct', 11: 'Nov', 12: 'Des'}
                                
                                for y in selected_years_comp:
                                    s_year = s_monthly[s_monthly.index.year == y]
                                    if not s_year.empty:
                                        # Use Month Numbers (1-12) for X to ensure correct sorting/stacking
                                        x_vals = s_year.index.month
                                        fig_comp.add_trace(go.Scatter(
                                            x=x_vals, y=s_year, name=f"{item} ({y})", mode='lines+markers',
                                            hovertemplate="%{y:.2f} kWh<extra></extra>"
                                        ))
                                
                                # Force X-Axis to show all 12 months with names
                                xaxis_args_comp = {
                                    'title': "Mes", 
                                    'tickmode': 'array',
                                    'tickvals': list(range(1, 13)),
                                    'ticktext': list(month_map.values())
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
                                        fig_comp.add_trace(go.Scatter(
                                            x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'
                                        ))
                                xaxis_args_comp = {'title': f"Dia ({target_month_name})", 'dtick': 1}

                            elif mode_t2 == 'Setmanal':
                                target_week = current_anchor.isocalendar().week
                                s_daily = s_full.resample('1d').sum()
                                
                                for y in selected_years_comp:
                                    mask = (s_daily.index.isocalendar().week == target_week) & (s_daily.index.year == y)
                                    s_sub = s_daily[mask]
                                    if not s_sub.empty:
                                        x_vals = s_sub.index.strftime('%a')
                                        fig_comp.add_trace(go.Scatter(
                                            x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'
                                        ))
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
                                        fig_comp.add_trace(go.Scatter(
                                            x=x_vals, y=s_sub, name=f"{item} ({y})", mode='lines+markers'
                                        ))
                                xaxis_args_comp = {'title': f"Hora ({target_day}/{target_month})", 'tickmode': 'linear', 'dtick': 1}

                        fig_comp.update_layout(title="Evolució Comparativa (Multianual)", xaxis=xaxis_args_comp, hovermode="x unified")
                        st.plotly_chart(fig_comp, use_container_width=True)

                        # SUMMARY TABLE (Tab 2)
                        st.subheader("Resum Series Seleccionades (Període/Anys)")
                        if selected_years_comp:
                            # Matrix Structure: Rows=Series, Cols=Years
                            data_matrix = []
                            
                            for it in selection_series:
                                row_dict = {"Sèrie": it}
                                
                                # Get Data
                                if it == "TOTAL": s_f, _ = get_aggregated_data(df, all_cups)
                                elif it == "ENLLUMENAT (Agregat)": s_f, _ = get_aggregated_data(df, lighting_cups)
                                elif it == "EDIFICIS (Agregat)": s_f, _ = get_aggregated_data(df, building_cups)
                                else:
                                     cols = df[it].columns
                                     ae_c = [c for c in cols if 'AE' in c and 'kWh' in c and 'AUTOCONS' not in c]
                                     if ae_c: s_f = df[it][ae_c[0]]
                                     else: continue
                                
                                # Calculate Per Year
                                total_row = 0
                                for y in selected_years_comp:
                                    mask_y = s_f.index.year == y
                                    val_y = s_f[mask_y].sum()
                                    row_dict[str(y)] = f"{val_y:.2f}"
                                    total_row += val_y
                                
                                # row_dict["Total Període"] = f"{total_row:.2f}" # Optional, if they said "no sum" maybe they don't want total? Let's keep specific years.
                                data_matrix.append(row_dict)
                            
                            st.dataframe(pd.DataFrame(data_matrix))
                            
                            # Bar Chart Visualization of the Summary
                            st.caption("Visualització Gràfica del Resum Anual")
                            fig_summ_bar = go.Figure()
                            
                            # We want Grouped Bar Chart: X=Series, Group=Year
                            # Iterate Years to create Traces
                            for y in selected_years_comp:
                                y_values = []
                                for row in data_matrix:
                                    # row has "Sèrie", "2023", "2024"...
                                    val_str = row.get(str(y), "0")
                                    val = float(val_str)
                                    y_values.append(val)
                                
                                x_names = [row["Sèrie"] for row in data_matrix]
                                
                                fig_summ_bar.add_trace(go.Bar(
                                    x=x_names,
                                    y=y_values,
                                    name=str(y),
                                    text=[f"{v:.0f}" for v in y_values],
                                    textposition='auto'
                                ))
                            
                            fig_summ_bar.update_layout(
                                barmode='group',
                                title="Comparativa de Totals per Sèrie i Any",
                                yaxis_title="kWh",
                                xaxis_title="Sèries",
                                legend_title="Any"
                            )
                            st.plotly_chart(fig_summ_bar, use_container_width=True)
                else:
                    st.info("Selecciona almenys una sèrie per visualitzar.")

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
                        fig_audit.add_trace(go.Scatter(x=day_series.index, y=day_series, name='Consum', fill='tozeroy', line=dict(color='#FF5733')))
                        
                        # Add background shading for Night
                        fig_audit.add_vrect(x0=day_series.index[0], x1=sunrise, fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Nit")
                        fig_audit.add_vrect(x0=sunset, x1=day_series.index[-1], fillcolor="black", opacity=0.1, layer="below", line_width=0, annotation_text="Nit")
                        
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
                    with st.spinner("Analitzant patrons complexos..."):
                        
                        # --- 1. NIGHT CONSUMPTION CHECK (BUILDINGS) ---
                        st.subheader("🌙 Consum Nocturn Elevat (Edificis)")
                        night_anomalies = []
                        
                        # Use all dates
                        unique_dates = sorted(list(set(df.index.date)))
                        
                        # Pre-calculate building list
                        # Only check if CUPS is in building_cups and NOT simply a default name if possible, 
                        # but building_cups is robust enough based on classification.
                        
                        for d in unique_dates:
                             day_data = df.loc[df.index.date == d]
                             
                             for cup in building_cups: # Only buildings
                                 cols = df[cup].columns
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
                                dd_n = df.loc[df.index.date == viewing_date_n]
                                cc_n = df[viewing_cup_n].columns
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
                                cols = df[c].columns
                                ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                                daily_sums = df[c][ae_col].resample('1d').sum() 
                                
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
                                                "Raw_CUPS": c,
                                                "Motiu": "Cap de Setmana/Festiu" if is_weekend else "Vacances (Agost/Nadal)",
                                                "Consum": f"{kwh_val:.2f} kWh",
                                                "Ref. Laborable": f"{baseline:.2f} kWh"
                                            })
                            
                            if school_anomalies:
                                df_school = pd.DataFrame(school_anomalies)
                                st.warning(f"S'han detectat {len(df_school)} dies amb consum alt en període inactiu.")
                                st.dataframe(df_school.drop(columns=['Raw_CUPS'], errors='ignore'))
                                
                                # Interactive Viewer for School
                                opt_sch = [f"{r['Data']} - {r['CUPS']} ({r['Motiu']})" for i, r in df_school.iterrows()]
                                sel_sch = st.selectbox("Veure Dia Escolar Anòmal", opt_sch)
                                if sel_sch:
                                    idx_s = opt_sch.index(sel_sch)
                                    r_s = df_school.iloc[idx_s]
                                    vd_s = r_s['Data']
                                    raw_cup_s = r_s['Raw_CUPS']
                                    
                                    # Plot
                                    dd_s = df.loc[df.index.date == vd_s]
                                    cc_s = df[raw_cup_s].columns
                                    ae_s_col = [x for x in cc_s if 'AE' in x and 'kWh' in x][0]
                                    series_s = dd_s[raw_cup_s][ae_s_col]
                                    
                                    fig_s = px.line(series_s, title=f"Consum Anòmal: {r_s['CUPS']} ({vd_s})", markers=True)
                                    fig_s.update_layout(xaxis=dict(dtick=3600000, tickformat="%H:%M"))
                                    st.plotly_chart(fig_s, use_container_width=True)
                            else:
                                st.success("✅ Les escoles semblen apagar correctament en festius.")

                        st.markdown("---")

                        # --- 3. SEASONAL PATTERN ANALYSIS ---
                        st.subheader("❄️☀️ Anàlisi Estacional (Hivern vs Estiu)")
                        
                        seasonal_data = []
                        for c in building_cups:
                            cols = df[c].columns
                            ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                            series = df[c][ae_col]
                            
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
                            cols = df[c].columns
                            ae_col = [k for k in cols if 'AE' in k and 'kWh' in k][0]
                            series = df[c][ae_col]
                            
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
if __name__ == "__main__":
    main()
