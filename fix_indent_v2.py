
import os

file_path = "app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
found_target = False
# Target line to start dedenting:
# Line 582 (1-based index) -> index 581
# Content: "            # If anchor is out of range, default to LATEST date (Max)"

start_index = 581
end_index = 1682 # Approximately end of main

for i, line in enumerate(lines):
    # Only dedent from the point where we see the error starts
    # We saw line 580 was correct (8 spaces), line 582 was incorrect (12 spaces)
    if i >= start_index and i < len(lines) - 2: # Keep the last 2 lines (__name__ check) intact
        if line.startswith("            "): # 12 spaces
            new_lines.append(line[4:]) # Remove 4 spaces
        elif line.startswith("            #"): # 12 spaces comment
            new_lines.append(line[4:])
        elif line.strip() == "": # Empty line
            new_lines.append(line)
        else:
            # If it doesn't have 12 spaces, maybe it's already correct or less indented?
            # If it has 8 spaces, leave it (it might be the end of the block?)
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Indentation fixed v2.")
