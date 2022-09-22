import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PRALINE')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)
    # data
    parser.add_argument('--data_path', default='/data/final/')
    parser.add_argument('--with_reformulations', action='store_true')

    # experiments
    parser.add_argument('--logs', default='experiments/logs', type=str)
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--checkpoint', default='', type=str)

    # task
    parser.add_argument('--task', default='multitask', choices=['multitask',
                                                                'domain_identification',
                                                                'ranking',
                                                                'verbalization'], type=str)

    # domain
    parser.add_argument('--domain', default='all', choices=['all',
                                                            'music',
                                                            'movies',
                                                            'books',
                                                            'tv_series',
                                                            'soccer'], type=str)

    # model
    parser.add_argument('--emb_dim', default=768, type=int)
    parser.add_argument('--in_dim', default=768, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--answer_max_length', default=50, type=int)
    parser.add_argument('--question_max_length', default=150, type=int)
    parser.add_argument('--pretrained_model', default='bert-base-uncased', type=str)

    # train
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--mismatch', default=0.8, type=int)
    parser.add_argument('--loss_ratio', default=0.99, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=10, type=int)
    parser.add_argument('--min_val_epoch', default=84, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--domain_weight', default=0.25, type=float)
    parser.add_argument('--ranking_weight', default=1.0, type=float)
    parser.add_argument('--verb_weight', default=0.25, type=float)

    return parser
