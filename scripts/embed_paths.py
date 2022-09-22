import os
import json
import h5py
import torch
import flair
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, BertEmbeddings

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('--data_path', default=str(ROOT_PATH.parent) + '/data/final', help='Data path')
parser.add_argument('--partition', default='test', choices=['train', 'val', 'test'], help='Partition')
parser.add_argument('--model', default='bert-base-uncased', help='Pretrained model')
args = parser.parse_args()

# read paths
paths = {}
with open(f'{args.data_path}/{args.partition}/paths.json') as json_file:
    paths = json.load(json_file)

# read labels dictionary for test set
labels_dict = {}
with open(f'{str(ROOT_PATH.parent)}/data/labels_dict.json') as json_file:
    labels_dict = json.load(json_file)

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
flair.device = DEVICE

# load bert model
pretrained_model = DocumentPoolEmbeddings([BertEmbeddings(args.model, layers='-1', pooling_operation='mean')])

# create embeddings
HDF5_DIR = f'{args.data_path}/{args.partition}/{args.model}_paths.h5'
with h5py.File(HDF5_DIR, 'w') as h5f:
    for startpoint, paths in tqdm(paths.items()):
        embeddings = []
        for path in paths:
            path_sentence = '[CLS] '
            mid = path[1]
            assert type(mid) is list
            for i, m in enumerate(mid):
                if not m.startswith('P') or m.split('-')[0] not in labels_dict:
                    continue

                predicate = m.split('-')[0]
                sep_token = ' [SEP] ' if i > 0 else ' '
                path_sentence = path_sentence + sep_token + labels_dict[predicate]

            flair_sentence = Sentence(path_sentence.lower())
            pretrained_model.embed(flair_sentence)
            embeddings.append(flair_sentence.embedding.detach().cpu().tolist())

        if embeddings:
            # save values
            h5f.create_dataset(name=startpoint, data=np.vstack(embeddings), compression="gzip", compression_opts=9)
