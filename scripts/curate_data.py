import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from dateutil.parser import parse

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='data path')
parser.add_argument('--partition', default='test', choices=['train', 'val', 'test'], help='parrtition to fix')
parser.add_argument('--final_path', default=str(ROOT_PATH.parent) + '/data/final', help='Final data path')
args = parser.parse_args()

# read conversation data
data = []
with open(f'{args.data_path}/{args.partition}/old_conversations.json') as json_file:
    data = json.load(json_file)

# read CONQUER startpoints
startpoints = {}
with open(f'{args.final_path}/{args.partition}/startpoints.json') as json_file:
    startpoints.update(json.load(json_file))

# read kg paths from CONQUER
paths = {}
with open(f'{args.final_path}/{args.partition}/paths.json') as json_file:
    paths.update(json.load(json_file))

def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False

def is_timestamp(string):
    if 'T00' not in string:
        return False
    return is_date(string)

# convert timestamp to dataset date format
def convert_timestamp(timestamp):
    date_time = parse(timestamp)
    return str(date_time.day) + " " + str(date_time.strftime('%B')) + " " + str(date_time.year)

# merge data
final_data = []
for d in tqdm(data):
    id = d['id']
    gold_paths = {}
    gold_paths_idx = {}

    for startpoint in d['startpoints']:
        for answer, answer_text in zip(d['answer'], d['answer_text']):
            # quantitative case
            if answer.isnumeric() and int(answer) > 3 and len(answer) < 4:
                startpoint_paths_mid = [p[1] for p in paths[startpoint]]
                no_qualifiers_path_mid = []
                for spm in startpoint_paths_mid:
                    no_qualifiers = []
                    for m in spm:
                        no_qualifiers.append(m.split('-')[0])
                    no_qualifiers_path_mid.append(' '.join(no_qualifiers))
                # case where we count the the paths with same middle part
                filter_mid_paths = [k for k, v in Counter(no_qualifiers_path_mid).items() if v == int(answer)]
                for fmp in filter_mid_paths:
                    for idx, q_path in enumerate(paths[startpoint]):
                        for mp in q_path[1]:
                            if fmp in mp:
                                if startpoint not in gold_paths:
                                    gold_paths[startpoint] = []
                                    gold_paths_idx[startpoint] = []
                                gold_paths[startpoint].append(q_path)
                                gold_paths_idx[startpoint].append(idx)
                if not gold_paths:
                    # case where we search for exact number at the end of the path
                    for idx, q_path in enumerate(paths[startpoint]):
                        if q_path[-1] == answer or q_path[-1] == f'+{answer}':
                            if startpoint not in gold_paths:
                                gold_paths[startpoint] = []
                                gold_paths_idx[startpoint] = []
                            gold_paths[startpoint].append(q_path)
                            gold_paths_idx[startpoint].append(idx)
            else:
                for idx, path in enumerate(paths[startpoint]):
                    # timestamp case
                    if is_date(answer) and is_timestamp(path[-1]) and (convert_timestamp(path[-1]).lower() == convert_timestamp(answer).lower() or str(parse(path[-1]).year) == answer):
                        if startpoint not in gold_paths:
                            gold_paths[startpoint] = []
                            gold_paths_idx[startpoint] = []
                        gold_paths[startpoint].append(path)
                        gold_paths_idx[startpoint].append(idx)
                    # entity (set) case
                    elif path[-1].lower() == answer.lower() or path[-1].lower() == answer_text.lower():
                        if startpoint not in gold_paths:
                            gold_paths[startpoint] = []
                            gold_paths_idx[startpoint] = []
                        gold_paths[startpoint].append(path)
                        gold_paths_idx[startpoint].append(idx)

    d['gold_paths'] = gold_paths
    d['gold_paths_idx'] = gold_paths_idx

print(f'{args.partition} length: {len(data)}')
with open(f'{args.final_path}/{args.partition}/conversations.json', 'w') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)
