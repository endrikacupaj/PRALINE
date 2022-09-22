import os
import json
from re import T
import h5py
import flair
import torch
from tqdm import tqdm
from pathlib import Path
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, BertEmbeddings

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

with open(f'{ROOT_PATH.parent}/data/final/domains.json', 'r') as json_file:
    domains = list(set(json.load(json_file).values()))

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
flair.device = DEVICE

# load bert model
pretrained_model = DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased', layers='-1', pooling_operation='mean')])

# create embeddings
embeddings = []
HDF5_DIR = f'{ROOT_PATH.parent}/data/final/domains.h5'
with h5py.File(HDF5_DIR, 'w') as h5f:
    for domain in tqdm(domains):
        flair_sentence = Sentence(domain.lower())
        pretrained_model.embed(flair_sentence)
        # save values
        h5f.create_dataset(name=domain, data=flair_sentence.embedding.detach().cpu().tolist(), compression="gzip", compression_opts=9)
