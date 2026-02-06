import pandas as pd
from datetime import timedelta

def classify_cups_by_name(df):
    """
    Classifies CUPS into 'Building' or 'Public Lighting' based on Name.
    Public Lighting: Starts with 'Enll' (case insensitive).
    """
    cups_list = df.columns.get_level_values(0).unique()
    lighting = []
    buildings = []
    
    for cups in cups_list:
        if str(cups).lower().startswith("enll"):
            lighting.append(cups)
        else:
            buildings.append(cups)
    return lighting, buildings

def detect_self_consumption_cups(df):
    """Identifies CUPS that have 'AE_AUTOCONS' (Self-Consumption) columns."""
    self_consumers = []
    for c in df.columns.get_level_values(0).unique():
        cols = df[c].columns
        if any('AUTOCONS' in col_var for col_var in cols):
             self_consumers.append(c)
    return self_consumers

def get_date_range(view_mode, anchor_date):
    """Returns (start_date, end_date, freq_alias) based on view mode and anchor."""
    start_date = None
    end_date = None
    freq = 'h'
    
    if view_mode == 'Diària':
        start_date = anchor_date
        end_date = anchor_date
        freq = '1h'
    elif view_mode == 'Setmanal':
        start_date = anchor_date - timedelta(days=anchor_date.weekday())
        end_date = start_date + timedelta(days=6)
        freq = '1d'
    elif view_mode == 'Mensual':
        start_date = anchor_date.replace(day=1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        freq = '1d'
    elif view_mode == 'Anual':
        start_date = anchor_date.replace(month=1, day=1)
        end_date = anchor_date.replace(month=12, day=31)
        freq = 'ME'
        
    return start_date, end_date, freq

def shift_date(view_mode, anchor_date, direction):
    """Shifts the anchor date forward or backward."""
    if view_mode == 'Diària':
        return anchor_date + timedelta(days=direction)
    elif view_mode == 'Setmanal':
        return anchor_date + timedelta(weeks=direction)
    elif view_mode == 'Mensual':
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
