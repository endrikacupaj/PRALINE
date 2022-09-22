import os
import json
import glob
from tqdm import tqdm
from pathlib import Path

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

data = []
for path in glob.glob(f'{ROOT_PATH.parent}/data/VerbalConvRef/*'):
    with open(path) as json_file:
        data.extend(json.load(json_file))

domain_dict = {}
for d in tqdm(data):
    domain = d['domain']
    domain_dict[d['conv_id']] = domain
    for q in d['questions']:
        domain_dict[q['question_id']] = domain
        for r in q['reformulations']:
            domain_dict[r['ref_id']] = domain

print(f'Domains: {list(set(domain_dict.values()))}')
with open(f'{ROOT_PATH.parent}/data/final/domains.json', 'w') as json_file:
    json.dump(domain_dict, json_file, ensure_ascii=False, indent=4)
