import re
import nltk
import torch
import sklearn
import torch.nn as nn

# import constants
from constants import *

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DomainIdentificationMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.scores = {
            ACCURACY: 0,
            PRECISION: 0,
            RECALL: 0,
            F1_SCORE: 0
        }

    def update(self, true, pred):
        self.y_true.append(true)
        self.y_pred.append(pred)
        self.scores = {
            ACCURACY: sklearn.metrics.accuracy_score(self.y_true, self.y_pred),
            PRECISION: sklearn.metrics.precision_score(self.y_true, self.y_pred, average='weighted'),
            RECALL: sklearn.metrics.recall_score(self.y_true, self.y_pred, average='weighted'),
            F1_SCORE: sklearn.metrics.f1_score(self.y_true, self.y_pred, average='weighted')
        }

def save_checkpoint(state, path):
    if args.with_reformulations:
        filename = f'{path}/with_reformulations_epoch_{state["epoch"]}_loss_{state["val_loss"]:.4f}.pth.tar'
    else:
        filename = f'{path}/without_reformulations_epoch_{state["epoch"]}_loss_{state["val_loss"]:.4f}.pth.tar'
    torch.save(state, filename)

class DomainIdentificationLoss(nn.Module):
    '''Domain Identification loss'''
    def __init__(self):
        super(DomainIdentificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output[DOMAIN_IDENTIFICATION], target[DOMAIN_IDENTIFICATION])

class RankingLoss(nn.Module):
    '''Ranking Loss'''
    def __init__(self):
        super(RankingLoss, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=args.margin).to(DEVICE)

    def forward(self, output, target):
        return self.criterion(output[RANKING][CONVERSATION], output[RANKING][PATH], target[RANKING])

class VerbalizationLoss(nn.Module):
    '''Answer Verbalization Loss'''
    def __init__(self):
        super(VerbalizationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, output, target):
        return self.criterion(output[VERBALIZATION], target[VERBALIZATION])

class MultitaskLoss(nn.Module):
    '''Multitask Learning Loss'''
    def __init__(self):
        super(MultitaskLoss, self).__init__()
        self.rank_loss = RankingLoss()
        self.domain_loss = DomainIdentificationLoss()
        self.verb_loss = VerbalizationLoss()

    def forward(self, output, target):
        losses = torch.stack(
            (
                self.domain_loss(output, target) * args.domain_weight,
                self.rank_loss(output, target) * args.ranking_weight,
                self.verb_loss(output, target) * args.verb_weight
            )
        )

        return {
            DOMAIN_IDENTIFICATION: losses[0],
            RANKING: losses[1],
            VERBALIZATION: losses[2],
            MULTITASK: losses.mean()
        }

def accuracy(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)

def precision(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred, average='weighted'),

def recall(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, average='weighted')

def f1_score(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

def bleu_score(reference, hypothesis):
    return {
        BLEU_SCORE_1: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1.0, 0.0, 0.0, 0.0)),
        BLEU_SCORE_2: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0.0, 0.0)),
        BLEU_SCORE_3: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0.0)),
        BLEU_SCORE_4: nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)),
    }

def meteor_score(reference, hypothesis):
    return nltk.translate.meteor_score.single_meteor_score(' '.join(reference), ' '.join(hypothesis))

def prec_at_1(predicted, actual):
    return torch.sum(torch.eq(predicted.topk(k=1)[1], actual)).item() / actual.size(0)

def hits_at_k(predicted, actual, k=10):
    if k > len(predicted): k = len(predicted)
    return torch.sum(torch.eq(predicted.topk(k=k)[1], actual)).item() / actual.size(0)

def mean_rank(predicted, actual):
    return torch.sum(torch.eq(predicted.argsort(descending=True), actual).nonzero().float().add(1.0)).item() / actual.size(0)

def mean_reciprocal_rank(predicted, actual):
    return torch.sum((1.0 / torch.eq(predicted.argsort(descending=True), actual).nonzero().float().add(1.0))).item() / actual.size(0)

def cover_answer(text, ans_ent=ANSWER_REGEX, ans_token=ANS_TOKEN):
    try:
        return re.sub(ans_ent, ans_token, text)
    except:
        return text

#check for existential question
def isExistential(question_start):
    existential_keywords = ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'have', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False

def convert_ids_to_string(tokenizer, ids):
    return tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        ).lower().replace("?", " ?").replace(".", " .").replace(",", " ,").replace("'", " '").split()

def tokenize_string(tokenizer, string, max_length):
    return tokenizer(string,
                    truncation=True,
                    padding=MAX_LENGTH,
                    max_length=max_length,
                    return_tensors=PT)[INPUT_IDS].to(DEVICE)
