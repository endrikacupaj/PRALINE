import os
import json
import h5py
import torch
import flair
import argparse
from tqdm import tqdm
from pathlib import Path
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, BertEmbeddings

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
parser.add_argument('--partition', default='train', choices=['train', 'val', 'test'], help='Partition')
parser.add_argument('--model', default='bert-base-uncased', help='Pretrained model')
args = parser.parse_args()

# read data
data = []
with open(f'{args.data_path}/{args.partition}/conversations.json') as json_file:
    data = json.load(json_file)

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)
flair.device = DEVICE

# load bert model
pretrained_model = DocumentPoolEmbeddings([BertEmbeddings(args.model, layers='-1', pooling_operation='mean')])

def embed(id, conv):
    flair_sentence = Sentence(conv.lower())
    pretrained_model.embed(flair_sentence)
    emb = flair_sentence.embedding.detach().cpu().tolist()

    # save values
    h5f.create_dataset(name=id, data=emb, compression="gzip", compression_opts=9)

# create embeddings
HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_conversations.h5'
with h5py.File(HDF5_DIR, 'w') as h5f:
    for d in tqdm(data):
        id = d['id']
        conv = d['all_conv']
        embed(id, conv)

        # embed reformulations with conversational history
        conv_history_list = conv.split('[SEP]')[:-1]
        for ref in d['reformulations']:
            ref_id = ref['ref_id']
            ref_question = ref['question']
            ref_conv = '[SEP]'.join(conv_history_list) + f'[SEP] {ref_question}'
            embed(ref_id, ref_conv)
