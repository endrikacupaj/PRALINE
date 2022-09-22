import random
import torch
import logging
import numpy as np
from tqdm import tqdm
from model import PRALINE
import torch.nn.functional as F
from data.conv_data import ConvDataset

from utils import *

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{str(ROOT_PATH)}/{args.logs}/test_wref_{args.domain}.log' if args.with_reformulations else f'{str(ROOT_PATH)}/{args.logs}/test_woref_{args.domain}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main():
    # prepare test data
    test_data = ConvDataset(TEST)

    # define model
    model = PRALINE(test_data.domains)

    model_path = f'{ROOT_PATH}/{args.snapshots}/{args.checkpoint}'
    logger.info(f"=> loading checkpoint '{model_path}'")
    checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    logger.info(f"=> loaded checkpoint '{model_path}' (epoch {checkpoint['epoch']})")

    # test_results = evaluate_ranking(test_data, model)
    # switch to evaluate mode
    model.eval()

    all_learned_paths = {}
    metrics = {
        DOMAIN_IDENTIFICATION: DomainIdentificationMeter(),
        RANKING: {
            PREC_AT_1: AverageMeter(),
            HITS_AT_5: AverageMeter(),
            HITS_AT_10: AverageMeter(),
            MR: AverageMeter(),
            MRR: AverageMeter()
        },
        VERBALIZATION: {
            BLEU_SCORE_1: AverageMeter(),
            BLEU_SCORE_2: AverageMeter(),
            BLEU_SCORE_3: AverageMeter(),
            BLEU_SCORE_4: AverageMeter(),
            METEOR_SCORE: AverageMeter()
        }
    }
    for i, conversation in enumerate(tqdm(test_data.conversations)):
        id = conversation[ID]

        if args.domain != 'all' and args.domain != test_data.domain_dict[id]:
            continue

        gold_startpoints = list(conversation[GOLD_PATHS_IDX].keys())

        # https://github.com/magkai/CONQUER/blob/master/main/rlEval.py#L414
        question_start = conversation[QUESTION].split(' ')[0].lower()
        if isExistential(question_start):
            # always yes
            metrics[RANKING][HITS_AT_5].update(1)
            metrics[RANKING][HITS_AT_10].update(1)
            if conversation[ANSWER_TEXT] == 'yes':
                metrics[RANKING][PREC_AT_1].update(1)
                metrics[RANKING][MR].update(1)
                metrics[RANKING][MRR].update(1)
            else:
                metrics[RANKING][PREC_AT_1].update(0)
                metrics[RANKING][MR].update(2)
                metrics[RANKING][MRR].update(0.5)

            continue
        elif not gold_startpoints:
            # skip for now
            metrics[RANKING][PREC_AT_1].update(0)
            metrics[RANKING][HITS_AT_5].update(0)
            metrics[RANKING][HITS_AT_10].update(0)
            # metrics[RANKING][MR].update(min(mr))
            metrics[RANKING][MRR].update(0)
            continue

        # temp metrics
        temp_domain_score = DomainIdentificationMeter()
        bleu_1, bleu_2, bleu_3, bleu_4, meteor = [], [], [], [], []
        prec1, hit5, hit10, mr, mrr = [], [], [], [], []

        def log_ranking(similarities):
            for gold_idx in conversation[GOLD_PATHS_IDX][startpoint]:
                tensor_gold_idx = torch.LongTensor([gold_idx]).to(DEVICE)
                prec1.append(prec_at_1(similarities, tensor_gold_idx))
                hit5.append(hits_at_k(similarities, tensor_gold_idx, 5))
                hit10.append(hits_at_k(similarities, tensor_gold_idx, 10))
                mr.append(mean_rank(similarities, tensor_gold_idx))
                mrr.append(mean_reciprocal_rank(similarities, tensor_gold_idx))

        def log_verbalization(reference, hypothesis):
            bleu = bleu_score(reference, hypothesis)
            bleu_1.append(bleu[BLEU_SCORE_1])
            bleu_2.append(bleu[BLEU_SCORE_2])
            bleu_3.append(bleu[BLEU_SCORE_3])
            bleu_4.append(bleu[BLEU_SCORE_4])
            meteor.append(meteor_score(reference, hypothesis))

        # tokenize conversation
        conversation_ids = tokenize_string(test_data.tokenizer, conversation[ALL_CONV].replace(START_TOKEN, '').replace(CTX_TOKEN, '').replace(SEP_TOKEN, '<sep>').strip(), args.question_max_length)
        # tokenize verbalized answer
        answer_ids = tokenize_string(test_data.tokenizer, cover_answer(conversation[VERBALIZED_ANSWER]), args.answer_max_length)

        # domain
        domain = test_data.domain_dict[id]
        domain_idx = test_data.domains.index(domain)
        embedded_domain = test_data.domain_emb[domain_idx].unsqueeze(0).to(DEVICE)

        # encoder output
        encoder_out = model.verbalization_module(conversation_ids, answer_ids)[ENCODER_OUT]

        # predict domain
        predicted_domain_idx = model.domain_pointer(encoder_out.max(1).values.unsqueeze(1)).argmax().item()

        # log domain results
        temp_domain_score.update(domain_idx, predicted_domain_idx)

        # conversation - domain representaion
        conv_domain_emb = model.ranking_module.learn_conv_domain(torch.cat([encoder_out.max(1).values, embedded_domain], dim=-1)).squeeze(0)

        # verbalization
        prediction_ids = model.verbalization_module.predict(conversation_ids)
        reference = convert_ids_to_string(test_data.tokenizer, answer_ids.squeeze())
        hypothesis = convert_ids_to_string(test_data.tokenizer, prediction_ids)

        # log verbalization results
        log_verbalization(reference, hypothesis)

        learned_paths = []
        for startpoint in gold_startpoints:
            if startpoint not in all_learned_paths:
                startpoint_paths = torch.from_numpy(test_data.path_emb[startpoint][()]).float().to(DEVICE)

                # paths representations
                all_learned_paths[startpoint] = model.ranking_module.learn_path(startpoint_paths).squeeze(1)

            learned_paths = all_learned_paths[startpoint]

            # conv2path similarities
            similarities = F.cosine_similarity(conv_domain_emb, learned_paths)

            # log ranking results
            log_ranking(similarities)

            if args.with_reformulations:
                # go through reformulations
                for ref in conversation[REFORMULATIONS]:
                    conv_history_list = conversation[ALL_CONV].split(SEP_TOKEN)[:-1]
                    new_all_conv = SEP_TOKEN.join(conv_history_list) + f'{SEP_TOKEN} {ref[QUESTION]} {CTX_TOKEN}'.replace(START_TOKEN, '').replace(CTX_TOKEN, '').replace(SEP_TOKEN, '<sep>').strip()
                    verbalized_answer = ref[VERBALIZED_ANSWER]

                    # tokenize reformulated conversation
                    conversation_ids = tokenize_string(test_data.tokenizer, new_all_conv, args.question_max_length)
                    # tokenize reformulated verbalized answer
                    answer_ids = tokenize_string(test_data.tokenizer, cover_answer(verbalized_answer), args.answer_max_length)

                    # encoder output
                    encoder_out = model.verbalization_module(conversation_ids, answer_ids)[ENCODER_OUT]

                    # conversation - domain representaion
                    conv_domain_emb = model.ranking_module.learn_conv_domain(torch.cat([encoder_out.max(1).values, embedded_domain], dim=-1)).squeeze(0)

                    # predict domain
                    predicted_domain_idx = model.domain_pointer(encoder_out.max(1).values.unsqueeze(1)).argmax().item()

                    # log domain results
                    temp_domain_score.update(domain_idx, predicted_domain_idx)

                    # conv2path similarities
                    similarities = F.cosine_similarity(conv_domain_emb, learned_paths) # for conv2path

                    # log ranking results
                    log_ranking(similarities)

                    # verbalization
                    prediction_ids = model.verbalization_module.predict(conversation_ids)
                    reference = convert_ids_to_string(test_data.tokenizer, answer_ids.squeeze())
                    hypothesis = convert_ids_to_string(test_data.tokenizer, prediction_ids)

                    # log verbalization results
                    log_verbalization(reference, hypothesis)

        # update domain pointer results
        metrics[DOMAIN_IDENTIFICATION].update(temp_domain_score.y_true[0], temp_domain_score.y_true[0] if temp_domain_score.y_true[0] in temp_domain_score.y_pred else temp_domain_score.y_pred[0])

        # update ranking results
        metrics[RANKING][PREC_AT_1].update(max(prec1))
        metrics[RANKING][HITS_AT_5].update(max(hit5))
        metrics[RANKING][HITS_AT_10].update(max(hit10))
        metrics[RANKING][MR].update(min(mr))
        metrics[RANKING][MRR].update(max(mrr))

        # update verbalization results
        metrics[VERBALIZATION][BLEU_SCORE_1].update(max(bleu_1))
        metrics[VERBALIZATION][BLEU_SCORE_2].update(max(bleu_2))
        metrics[VERBALIZATION][BLEU_SCORE_3].update(max(bleu_3))
        metrics[VERBALIZATION][BLEU_SCORE_4].update(max(bleu_4))
        metrics[VERBALIZATION][METEOR_SCORE].update(max(meteor))

        if (i+1) % 100 == 0:
            print_results(metrics)

    print_results(metrics)


def print_results(metrics):
    logger.info(f'Domain Identification Pointer:')
    logger.info(f'\tAccuracy: {metrics[DOMAIN_IDENTIFICATION].scores[ACCURACY]:.4f}')
    logger.info(f'\tPrecision: {metrics[DOMAIN_IDENTIFICATION].scores[PRECISION]:.4f}')
    logger.info(f'\tRecall: {metrics[DOMAIN_IDENTIFICATION].scores[RECALL]:.4f}')
    logger.info(f'\tF1 score: {metrics[DOMAIN_IDENTIFICATION].scores[F1_SCORE]:.4f}')

    logger.info(f'Ranking:')
    logger.info(f'\tPrec@1: {metrics[RANKING][PREC_AT_1].avg:.4f}')
    logger.info(f'\tHits@5: {metrics[RANKING][HITS_AT_5].avg:.4f}')
    logger.info(f'\tHits@10: {metrics[RANKING][HITS_AT_10].avg:.4f}')
    logger.info(f'\tMean Rank: {metrics[RANKING][MR].avg:.4f}')
    logger.info(f'\tMean Reciprocal Rank: {metrics[RANKING][MRR].avg:.4f}')

    logger.info(f'Verbalization:')
    logger.info(f'\tBLEU 1: {metrics[VERBALIZATION][BLEU_SCORE_1].avg:.4f}')
    logger.info(f'\tBLEU 2: {metrics[VERBALIZATION][BLEU_SCORE_2].avg:.4f}')
    logger.info(f'\tBLEU 3: {metrics[VERBALIZATION][BLEU_SCORE_3].avg:.4f}')
    logger.info(f'\tBLEU 4: {metrics[VERBALIZATION][BLEU_SCORE_4].avg:.4f}')
    logger.info(f'\tMETEOR: {metrics[VERBALIZATION][METEOR_SCORE].avg:.4f}')

if __name__ == '__main__':
    main()