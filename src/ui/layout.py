import streamlit as st
import datetime
from ..core.database import init_supabase

def render_login():
    """Renders the login form and handles authentication."""
    st.markdown("#### 🔐 Accés al Sistema")
    email = st.text_input("Correu Electrònic")
    password = st.text_input("Contrasenya", type="password")
    
    if st.button("Iniciar Sessió"):
        if not email or not password:
            st.error("Introdueix usuari i contrasenya")
            return
            
        supabase = init_supabase()
        if not supabase: return
        
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state.user = res.user
            st.session_state.session = res.session
            st.rerun()
        except Exception as e:
            st.error(f"Error d'accés: {e}")

def render_sidebar(show_source_selector=True):
    """Renders the sidebar with user info, logout, and global project controls."""
    if not st.session_state.get('user'):
        return None
        
    st.sidebar.write(f"👤 {st.session_state.user.email}")
    if st.sidebar.button("Tancar Sessió"):
        supabase = init_supabase()
        if supabase:
            supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()
        
    st.sidebar.divider()
    
    # Data Source Selection
    if show_source_selector:
        source = st.sidebar.radio(
            "Font de Dades", 
            ["Base de Dades (Supabase)", "Pujar CSV Local (Processat)", "Importar Edistribucion (Originals)"], 
            index=0
        )
        st.sidebar.divider()
        return source
    else:
        return "Base de Dades (Supabase)"

def init_session_state():
    """Initializes the required session state variables."""
    if 'selected_cups_list' not in st.session_state: st.session_state.selected_cups_list = []
    if 'anchor_date' not in st.session_state: st.session_state.anchor_date = datetime.date.today()
    if 'user' not in st.session_state: st.session_state.user = None
    if 'view_mode' not in st.session_state: st.session_state.view_mode = 'Anual'
    # Initialize keys used in app.py logic to 'Anual'
    if 'view_mode_t1_v2' not in st.session_state: st.session_state.view_mode_t1_v2 = 'Anual'
    if 'mode_t2' not in st.session_state: st.session_state.mode_t2 = 'Anual'
