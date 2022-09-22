import re
import json
import h5py
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import cover_answer

# import constants
from constants import *

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

class ConvDataset(Dataset):
    def __init__(self, partition):
        if partition is None or partition not in [TRAIN, VAL, TEST]:
            raise Exception(f'Unknown partition type {partition}.')
        else:
            self.partition = partition

        # read data
        self.conversations = json.load(open(f'{ROOT_PATH}/{args.data_path}/{partition}/conversations.json'))

        # read embedded data
        self.path_emb = h5py.File(f'{ROOT_PATH}/{args.data_path}/{partition}/{args.pretrained_model}_paths.h5', 'r')

        # read domain data
        self.domain_dict = json.load(open(f'{ROOT_PATH}/{args.data_path}/domains.json'))
        self.domains = list(set(self.domain_dict.values()))
        self.domain_emb = []
        with h5py.File(f'{ROOT_PATH}/{args.data_path}/domains.h5', 'r') as domain_h5:
            for domain in self.domains:
                self.domain_emb.append(torch.from_numpy(domain_h5[domain][()]).float())

        # for train and val, remove conversations without gold paths
        if partition in [TRAIN, VAL]:
            seen_convs = set()
            filtered_data = []
            for conv in self.conversations:
                all_conv = re.sub(r'\s+', '', conv[ALL_CONV].lower())
                if conv[GOLD_PATHS] and conv[GOLD_PATHS_IDX] and all_conv not in seen_convs:
                    filtered_data.append(conv)
                    seen_convs.add(all_conv)

            self.conversations = filtered_data

        self.mismatch = args.mismatch

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)

        # save tokenized utterances
        self.tokenized = {}

        unique_conv = []
        self.expended_data = []
        for conv in self.conversations:
            id = conv[ID]
            conv_data = conv[ALL_CONV].replace(START_TOKEN, '').replace(CTX_TOKEN, '').replace(SEP_TOKEN, '<sep>').strip()
            verbalized_answer = conv[VERBALIZED_ANSWER]

            # get domain
            domain = self.domain_dict[id]
            domain_idx = self.domains.index(domain)
            domain_emb = self.domain_emb[domain_idx]

            conversation_ids = self.tokenizer(conv_data,
                                        truncation=True,
                                        padding=MAX_LENGTH,
                                        max_length=args.question_max_length,
                                        return_tensors=PT)[INPUT_IDS].squeeze()

            answer_ids = self.tokenizer(cover_answer(verbalized_answer),
                                        truncation=True,
                                        padding=MAX_LENGTH,
                                        max_length=args.answer_max_length,
                                        return_tensors=PT)[INPUT_IDS].squeeze()
            if conv_data.replace(' ', '').lower() not in unique_conv:
                self.expended_data.append({
                    ID: id,
                    DOMAIN_IDS: domain_idx,
                    DOMAIN_EMB: domain_emb,
                    GOLD_PATHS_IDX: conv[GOLD_PATHS_IDX],
                    CONVERSATION_IDS: conversation_ids,
                    ANSWER_IDS: answer_ids
                })
                unique_conv.append(conv_data.replace(' ', '').lower())
            if args.with_reformulations:
                for reformulation_data in conv[REFORMULATIONS]:
                    id = reformulation_data[REF_ID]
                    conv_history_list = conv_data.split(SEP_TOKEN)[:-1]
                    conv_data = SEP_TOKEN.join(conv_history_list) + f'{SEP_TOKEN} {reformulation_data[QUESTION]}'.replace(START_TOKEN, '').replace(CTX_TOKEN, '').replace(SEP_TOKEN, '<sep>').strip()
                    verbalized_answer = reformulation_data[VERBALIZED_ANSWER]

                    conversation_ids = self.tokenizer(conv_data,
                                            truncation=True,
                                            padding=MAX_LENGTH,
                                            max_length=args.question_max_length,
                                            return_tensors=PT)[INPUT_IDS].squeeze()

                    answer_ids = self.tokenizer(cover_answer(verbalized_answer),
                                                truncation=True,
                                                padding=MAX_LENGTH,
                                                max_length=args.answer_max_length,
                                                return_tensors=PT)[INPUT_IDS].squeeze()
                    if conv_data.replace(' ', '').lower() not in unique_conv:
                        self.expended_data.append({
                            ID: id,
                            DOMAIN_IDS: domain_idx,
                            DOMAIN_EMB: domain_emb,
                            GOLD_PATHS_IDX: conv[GOLD_PATHS_IDX],
                            CONVERSATION_IDS: conversation_ids,
                            ANSWER_IDS: answer_ids
                        })
                        unique_conv.append(conv_data.replace(' ', '').lower())

    def __getitem__(self, index):
        conversation_data = self.expended_data[index]

        # select random startpoint
        startpoint, gold_path_idx = random.choice(list(conversation_data[GOLD_PATHS_IDX].items()))

        # we force a mismatch given the probability
        match = np.random.uniform() > self.mismatch if self.partition == TRAIN else True

        target = match and 1 or -1

        if target == 1: # load positive samples
            path_idx = random.choice(gold_path_idx)
            path = torch.from_numpy(self.path_emb[startpoint][()][path_idx]).float()
        else:
            # Negative samples are generated by picking random false path
            all_idx = range(len(self.path_emb[startpoint][()]))
            random_idx = random.choice(all_idx)
            # random index to pick path
            while random_idx in gold_path_idx:
                random_idx = random.choice(all_idx) # pick a random index

            # load negative samples
            path = torch.from_numpy(self.path_emb[startpoint][()][random_idx]).float()

        return {
            ID: conversation_data[ID],
            DOMAIN_IDS: conversation_data[DOMAIN_IDS],
            DOMAIN_EMB: conversation_data[DOMAIN_EMB],
            PATH: path,
            RANKING_TARGET: target,
            CONVERSATION_IDS: conversation_data[CONVERSATION_IDS],
            ANSWER_IDS: conversation_data[ANSWER_IDS]
        }

    def __len__(self):
        return len(self.expended_data)
