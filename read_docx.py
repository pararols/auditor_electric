
import zipfile
import re
import sys

docx_path = r"c:\Users\parar\OneDrive\Documents\antigravity\auditor electric\huawei\Huawei API Error 407 Troubleshooting.docx"

try:
    with zipfile.ZipFile(docx_path) as z:
        if 'word/document.xml' not in z.namelist():
            print("Error: word/document.xml not found in zip.")
            sys.exit(1)
            
        xml_content = z.read('word/document.xml').decode('utf-8')
        
        # XML parsing is safer than regex for full text but regex is quicker for a snippet
        # Let's use a slightly better regex approach or just dump it.
        # Removing all tags
        text = re.sub('<[^>]+>', ' ', xml_content)
        
        # Collapse whitespace
        text = ' '.join(text.split())
        
        print("--- START OF DOCX CONTENT ---")
        print(text)
        print("--- END OF DOCX CONTENT ---")

except Exception as e:
    print(f"Error processing docx: {e}")
