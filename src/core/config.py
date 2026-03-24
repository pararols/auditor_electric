import streamlit as st

# --- Page Config Defaults ---
PAGE_TITLE = "Auditor Energètic & Enllumenat Públic"
PAGE_ICON = "⚡"

# --- CUPS Mappings ---
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
    # Participants Privats CLE Sala Nova (Anonimitzats)
    "ES0031406231815001DS0F": "Part. Privat 01",
    "ES0031408273101001WC0F": "Part. Privat 02",
    "ES0031406053483001CX0F": "Part. Privat 03",
    "ES0031408467847001TY0F": "Part. Privat 04",
    "ES0031406053551001DE0F": "Part. Privat 05",
    "ES003140863192001MC0F": "Part. Privat 06",
    "ES0031406053458001ZB0F": "Part. Privat 07",
    "ES0031406054122001LR0F": "Part. Privat 08",
    "ES0031406053565001CS0F": "Part. Privat 09",
    "ES0031406053376001ZY0F": "Part. Privat 10"
}

# --- Community Participants Whitelist ---
COMMUNITY_PARTICIPANTS = [
    "ES0031406053357001QG0F", # Sala Nova
    "ES0031406053560001XY0F", # Escola
    "ES0031406054170001JT0F", # Ajuntament
    "ES0031408303814001QQ0F", # Llar Infants
    "ES0031408332025001ZK0F", # Polivalent
    "ES0031408457126001XL0F"  # Pavelló
]

# Repartiment oficial CLE Sala Nova (Potència kWp i Coeficient de Repartiment)
# Inclou participants municipals i privats (Total 100%)
COMMUNITY_QUOTAS = {
    # Municipals
    "ES0031406053357001QG0F": {"equipament": "Sala Nova", "kwp": 0.3160935, "coef": 0.018485},
    "ES0031406053560001XY0F": {"equipament": "Escola", "kwp": 4.5844245, "coef": 0.268095},
    "ES0031406054170001JT0F": {"equipament": "Ajuntament", "kwp": 3.8617785, "coef": 0.225835},
    "ES0031408303814001QQ0F": {"equipament": "Llar d'Infants", "kwp": 1.1856114, "coef": 0.069334},
    "ES0031408332025001ZK0F": {"equipament": "Sala Polivalent", "kwp": 0.4567752, "coef": 0.026712},
    "ES0031408457126001XL0F": {"equipament": "Pavelló", "kwp": 1.6952769, "coef": 0.099139},
    # Privats (Veïns - Anonimitzats)
    "ES0031406231815001DS0F": {"equipament": "Part. Privat 01", "kwp": 0.500004, "coef": 0.029240},
    "ES0031408273101001WC0F": {"equipament": "Part. Privat 02", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406053483001CX0F": {"equipament": "Part. Privat 03", "kwp": 0.500004, "coef": 0.029240},
    "ES0031408467847001TY0F": {"equipament": "Part. Privat 04", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406053551001DE0F": {"equipament": "Part. Privat 05", "kwp": 0.500004, "coef": 0.029240},
    "ES003140863192001MC0F": {"equipament": "Part. Privat 06", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406053458001ZB0F": {"equipament": "Part. Privat 07", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406054122001LR0F": {"equipament": "Part. Privat 08", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406053565001CS0F": {"equipament": "Part. Privat 09", "kwp": 0.500004, "coef": 0.029240},
    "ES0031406053376001ZY0F": {"equipament": "Part. Privat 10", "kwp": 0.500004, "coef": 0.029240}
}

# --- Localization ---
month_names = {
    1: 'Gener', 2: 'Febrer', 3: 'Març', 4: 'Abril', 5: 'Maig', 6: 'Juny',
    7: 'Juliol', 8: 'Agost', 9: 'Setembre', 10: 'Octubre', 11: 'Novembre', 12: 'Desembre'
}
month_names_short = {
    1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 
    7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Oct', 11: 'Nov', 12: 'Des'
}

# --- UI Styling ---
def apply_custom_styles():
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
            border: 1px solid #e0e0e0;
        }
        /* Custom sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
