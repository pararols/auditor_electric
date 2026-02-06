
import os

file_path = "c:\\Users\\parar\\OneDrive\\Documents\\antigravity\\auditor electric\\app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
start_fix = 582 # 1-based in my view, so index 581.
# Actually let's check content to be sure.
# Line 583 in file viewer (1-based) was the error. 
# "            if not (min_csv_date <= st.session_state.anchor_date <= max_csv_date):"

# We want to dedent lines starting from index 581 (Line 582) down to the end of main function.
# The end of main is Line 1682.
# Lines 1683+ are "if __name__ == ..." which should be at column 0.

for i, line in enumerate(lines):
    line_idx = i + 1
    if 582 <= line_idx <= 1682:
        # Check if it has at least 4 spaces
        if line.startswith("    "):
            new_lines.append(line[4:])
        else:
            # If line is empty or shorter, just keep it (might be empty newline)
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Indentation fixed.")
