import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun

def detect_flux_regulation(df_cups, view_date=None):
    """
    Analitza la regulació de fluxe per a un CUPS en una data concreta (o mitjana nits recents).
    Calcula: Potència instal·lada (peak), Potència reduïda (plató), Horaris i % de reducció.
    """
    # Setup Location (Girona)
    city = LocationInfo("Girona", "Catalonia", "Europe/Madrid", 41.9, 2.8)
    
    if view_date is None:
        view_date = df_cups.index.max().date()
        
    s = sun(city.observer, date=view_date)
    sunrise = s['sunrise'].replace(tzinfo=None)
    sunset = s['sunset'].replace(tzinfo=None)
    
    # Agafem la nit (sunset fins sunrise de l'endemà o el mateix dia)
    # Per simplificar, analitzem les hores de foscor del dia natural
    mask_night = (df_cups.index.date == view_date) & ((df_cups.index < sunrise) | (df_cups.index > sunset))
    night_data = df_cups[mask_night]
    
    if night_data.empty:
        return None
        
    # P_top: Potència màxima (1h després de posta o abans de sortida)
    # P_bottom: El plató més baix de la nit
    p_max = night_data.max()
    p_min = night_data.min()
    
    # Filtrem soroll (només si p_max té sentit)
    if p_max < 0.1: return None
    
    reduction_ratio = 1 - (p_min / p_max) if p_max > 0 else 0
    
    # Horaris de regulació: hores on el consum està a prop de p_min (tol 10%)
    reg_mask = night_data <= (p_min * 1.15)
    reg_hours = night_data[reg_mask].index.hour.tolist()
    
    return {
        "p_installed": p_max,
        "p_reduced": p_min,
        "reduction_pct": reduction_ratio * 100,
        "hours": sorted(reg_hours),
        "is_regulated": reduction_ratio > 0.15 and len(reg_hours) >= 2
    }

def detect_historical_power_shifts(df_cups, min_change_pct=15):
    """
    Detecta canvis permanents en la potència nocturna al llarg del temps.
    Retorna llista de canvis detectats amb data, valors abans/després i reducció.
    """
    # 1. Calcular el "Plató Nocturn" diari (P95 de les hores de nit)
    # Simplificació: hores de 00h a 05h per evitar sunset/sunrise variable en càlcul massiu
    night_mask = (df_cups.index.hour >= 0) & (df_cups.index.hour <= 5)
    daily_night_plateau = df_cups[night_mask].resample('1d').quantile(0.95)
    
    # 2. Agregació mensual per suavitzar variacions i festius
    monthly_plateau = daily_night_plateau.resample('ME').median()
    
    shifts = []
    for i in range(1, len(monthly_plateau)):
        p_old = monthly_plateau.iloc[i-1]
        p_new = monthly_plateau.iloc[i]
        
        if pd.isna(p_old) or pd.isna(p_new) or p_old == 0: continue
        
        change_pct = (p_new - p_old) / p_old * 100
        
        if abs(change_pct) >= min_change_pct:
            shifts.append({
                "date": monthly_plateau.index[i].strftime('%m/%Y'),
                "p_old": p_old,
                "p_new": p_new,
                "change_kw": p_new - p_old,
                "change_pct": change_pct
            })
            
    return shifts
