import os
from pypdf import PdfReader

pdf_path = r"c:\Users\parar\OneDrive\Documents\antigravity\auditor electric\huawei\SmartPVMS 24.2.0 Northbound API Reference.pdf"

try:
    reader = PdfReader(pdf_path)
    print(f"Total Pages: {len(reader.pages)}")
    
    keywords = ["getKpiStationDay", "getKpiStationHour", "productPower", "inverter_power", "yield"]
    
    found_info = {}
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        for kw in keywords:
            if kw in text:
                if kw not in found_info:
                    found_info[kw] = []
                # Store a snippet
                idx = text.find(kw)
                start = max(0, idx - 200)
                end = min(len(text), idx + 500) # Get enough context for table definition
                found_info[kw].append(f"Page {i+1}: ...{text[start:end]}...")

    # Print results
    print("--- Analysis Results ---")
    for kw, snippets in found_info.items():
        print(f"\nKEYWORD: {kw}")
        for s in snippets[:3]: # Limit to first 3 matches
            print(s)
            print("-" * 40)
            
except Exception as e:
    print(f"Error reading PDF: {e}")
