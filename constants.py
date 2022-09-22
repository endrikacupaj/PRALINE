import os
import torch
from pathlib import Path
from args import get_parser

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# model name
MODEL_NAME = 'PRALINE'

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
torch.cuda.set_device(args.cuda_device)

# partition
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# tasks
MULTITASK = 'multitask'
RANKING = 'ranking'
VERBALIZATION = 'verbalization'
DOMAIN_IDENTIFICATION = 'domain_identification'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
NA_TOKEN = 'NA'
ENT_TOKEN = '[ENT]'
ANS_TOKEN = '[ANS]'

# training
TASK = 'task'
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'
LOSS = 'loss'
VAL_LOSS = 'val_loss'

# testing
PREC_AT_1 = 'prec@1'
HITS_AT_1 = 'hits@1'
HITS_AT_5 = 'hits@5'
HITS_AT_10 = 'hits@10'
MR = 'mr'
MRR = 'mrr'
BLEU_SCORE = 'bleu_score'
BLEU_SCORE_1 = 'bleu_score_1'
BLEU_SCORE_2 = 'bleu_score_2'
BLEU_SCORE_3 = 'bleu_score_3'
BLEU_SCORE_4 = 'bleu_score_4'
METEOR_SCORE = 'meteor_score'
ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1_SCORE = 'f1_score'

# other
ID = 'id'
CONVERSATION = 'conversation'
PATH = 'path'
BERT_BASE_UNCASED = 'bert-base-uncased'
BART_MODEL = 'facebook/bart-base'
ANSWER_REGEX = r'\[.*?\]'
RANKING_TARGET = 'ranking_target'
VERBALIZED_ANSWER = 'verbalized_answer'
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
CONVERSATION_IDS = 'conversation_ids'
ANSWER_IDS = 'answer_ids'
MAX_LENGTH = 'max_length'
PT = 'pt'
GOLD_PATHS = 'gold_paths'
GOLD_PATHS_IDX = 'gold_paths_idx'
ALL_CONV = 'all_conv'
CONVERSATION_UTTERANCE = 'conversation_utterance'
ANSWER_UTTERANCE = 'answer_utterance'
REFORMULATIONS = 'reformulations'
REF_ID = 'ref_id'
QUESTION = 'question'
ANSWER_TEXT = 'answer_text'
DOMAIN_IDS = 'domain_idx'
DOMAIN_EMB = 'domain_emb'
LOGITS = 'logits'
ENCODER_OUT = 'encoder_out'