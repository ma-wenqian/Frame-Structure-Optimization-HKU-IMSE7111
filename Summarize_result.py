import os
import re
import csv

results_dir = 'results'
output_csv = os.path.join(results_dir, 'result_summary.csv')

header = ['filename', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'TotalMass', 'max_u17']
rows = []

for fname in sorted(os.listdir(results_dir)):
    if  fname.endswith('.txt'):
        with open(os.path.join(results_dir, fname), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Find d array (may span multiple lines)
            d_str = ''
            d_found = False
            for line in lines:
                if 'd=[' in line:
                    d_str = line.strip()
                    d_found = True
                elif d_found and ']' not in d_str:
                    d_str += ' ' + line.strip()
                if d_found and ']' in d_str:
                    break
            d_values = []
            d_match = re.search(r'd=\[([0-9.eE+\-\s]+)\]', d_str)
            if d_match:
                d_values = d_match.group(1).split()
            # Find TotalMass and max_u17
            total_mass = ''
            max_u17 = ''
            for line in lines:
                tm_match = re.search(r'TotalMass=([0-9.eE+-]+)T, max_u17=([0-9.eE+-]+)mm', line)
                if tm_match:
                    total_mass = tm_match.group(1)
                    max_u17 = tm_match.group(2)
                    break
            if len(d_values) == 8:
                rows.append([fname] + d_values + [total_mass, max_u17])

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Summary written to {output_csv}")