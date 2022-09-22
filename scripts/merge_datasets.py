import os
import json
import glob
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

random.seed(1234)

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--verbal_data_path', default=str(ROOT_PATH.parent) + '/data/Verbal-ConvQuestions', help='Verbal-ConvQuestions path')
parser.add_argument('--convref_data_path', default=str(ROOT_PATH.parent) + '/data/ConvRef', help='ConvRef path')
parser.add_argument('--verbal_convref_data_path', default=str(ROOT_PATH.parent) + '/data/VerbalConvRef', help='VerbalConvRef path')
args = parser.parse_args()

# read verbalized data
train, val, test = [], [], []
for train_file in glob.glob(f'{args.verbal_data_path}/train/*.json'):
    with open(train_file) as json_file:
        train.extend(json.load(json_file))

for val_file in glob.glob(f'{args.verbal_data_path}/val/*.json'):
    with open(val_file) as json_file:
        val.extend(json.load(json_file))

for test_file in glob.glob(f'{args.verbal_data_path}/test/*.json'):
    with open(test_file) as json_file:
        test.extend(json.load(json_file))

va_dict = {}
# extract verbalized answers in dictionary
for part in [train, val, test]:
    for conv in part:
        for turn in conv['questions']:
            verbalized_answers = [turn['verbalized_answer']]
            for va in turn['paraphrased_answer']:
                verbalized_answers.append(va)
            va_dict[turn['question_id']] = verbalized_answers

# read convref data
train, val, test = [], [], []
with open(f'{args.convref_data_path}/ConvRef_trainset.json') as json_file:
    train.extend(json.load(json_file))

with open(f'{args.convref_data_path}/ConvRef_devset.json') as json_file:
    val.extend(json.load(json_file))

with open(f'{args.convref_data_path}/ConvRef_testset.json') as json_file:
    test.extend(json.load(json_file))

# merge data
final_train, final_test, final_val = [], [], []
for partition in tqdm([[train, final_train, 'train'], [val, final_val, 'val'], [test, final_test, 'test']]):
    for conv in tqdm(partition[0]):
        for t in conv['questions']:
            all_va = va_dict[t['question_id']]
            t['verbalized_answer'] = all_va[0]
            for ref in t['reformulations']:
                ref['question'] = ref.pop('reformulation')
                ref['verbalized_answer'] = random.choice(all_va)
                del ref['baseline_answer']

            t['answer'] = t.pop('gold_answer')
            t['answer_text'] = t.pop('gold_answer_text')
            del t['baseline_answer']

        partition[1].append(conv)

    print(f'{partition[2]} length: {len(partition[1])}')
    with open(f'{args.verbal_convref_data_path}/{partition[2]}.json', 'w') as json_file:
        json.dump(partition[1], json_file, ensure_ascii=False, indent=4)