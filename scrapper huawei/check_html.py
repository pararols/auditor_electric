
import os

files = ["debug_target.html"]
keywords = ["Trend", "Historical", "Reports", "Energy", "Overview", "Management", "Statistics", "Curve",
            "Informe", "Datos", "Histórico", "Exportar", "Tendencia", "Energía", "Gestión", "Planta",
            "Descargar", "Download", "CSV", "Excel", "Fecha", "Date", "Día", "Day", "Horario", "Hourly", "Hour", "Hora", "Time",
            "input", "calendar", "picker", "placeholder"]

for filename in files:
    if os.path.exists(filename):
        print(f"Checking {filename}...")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"  File size: {len(content)} characters")
                for kw in keywords:
                    count = content.count(kw)
                    print(f"  '{kw}': found {count} times")
                    if count > 0:
                        # Print context of first match
                        idx = content.find(kw)
                        start = max(0, idx - 50)
                        end = min(len(content), idx + 50)
                        print(f"    Context: ...{content[start:end]}...")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
    else:
        print(f"{filename} not found.")
