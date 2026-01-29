
import json

file_path = r'c:\Users\Administrator\workspace\seoul-auction-prediction\notebooks\seoul_auction_junseo2.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if '/mnt/user-data/outputs/' in line:
                new_line = line.replace('/mnt/user-data/outputs/', '../results/outputs/')
                new_source.append(new_line)
                changed = True
            else:
                new_source.append(line)
        cell['source'] = new_source

if changed:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully updated the notebook.")
else:
    print("No occurrences found to update.")
