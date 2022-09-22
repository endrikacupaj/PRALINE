import os
import time
import random
import torch
import logging
import numpy as np
from pathlib import Path
from tqdm.std import tqdm
from model import PRALINE
from transformers import AdamW
from data.conv_data import ConvDataset
from torch.utils.data import DataLoader
from utils import (AverageMeter, DomainIdentificationLoss, RankingLoss,
                    VerbalizationLoss, MultitaskLoss, save_checkpoint)

# import constants
from constants import *

# create directories for experiments
logging_path = f'{str(ROOT_PATH)}/{args.logs}'
Path(logging_path).mkdir(parents=True, exist_ok=True)
checkpoint_path = f'{str(ROOT_PATH)}/{args.snapshots}'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{logging_path}/train_wref.log' if args.with_reformulations else f'{logging_path}/train_woref.log', 'w'),
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
    # log arguments
    logger.info(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    # prepare training data
    train_data = ConvDataset(TRAIN)
    val_data = ConvDataset(VAL)
    test_data = ConvDataset(TEST)

    # define model
    model = PRALINE(train_data.domains).to(DEVICE)

    # define framework loss
    criterion = {
        DOMAIN_IDENTIFICATION: DomainIdentificationLoss,
        RANKING: RankingLoss,
        VERBALIZATION: VerbalizationLoss,
        MULTITASK: MultitaskLoss
    }[args.task]()

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # load checkpoint
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}''")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    # log num of params
    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # prepare training loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    logger.info('Training loader prepared.')

    # prepare validation loader
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    logger.info('Validation loader prepared.')

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_losses = validate(val_loader, model, criterion)
        logger.info(f'Val Domain loss: {val_losses[DOMAIN_IDENTIFICATION].avg:.4f} - Val Ranking loss: {val_losses[RANKING].avg:.4f} - Val Verbalization loss: {val_losses[VERBALIZATION].avg:.4f} - Multitask loss: {val_losses[MULTITASK].avg:.4f}')

        # save the model
        if (epoch+1) % args.valfreq == 0 or (epoch+1) > args.min_val_epoch:
            save_checkpoint({
                EPOCH: epoch + 1,
                STATE_DICT: model.state_dict(),
                OPTIMIZER: optimizer.state_dict(),
                VAL_LOSS: val_losses[MULTITASK].avg},
                path=checkpoint_path)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = {
        RANKING: AverageMeter(),
        DOMAIN_IDENTIFICATION: AverageMeter(),
        VERBALIZATION: AverageMeter(),
        MULTITASK: AverageMeter()
    }

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # extract batch data
        domain_ids = batch[DOMAIN_IDS].to(DEVICE)
        domain_emb = batch[DOMAIN_EMB].to(DEVICE)
        paths = batch[PATH].to(DEVICE)
        ranking_target = batch[RANKING_TARGET].to(DEVICE)
        conversation_ids = batch[CONVERSATION_IDS].to(DEVICE)
        answer_ids = batch[ANSWER_IDS].to(DEVICE)

        # input
        input = {
            CONVERSATION_IDS: conversation_ids,
            ANSWER_IDS: answer_ids,
            DOMAIN_EMB: domain_emb,
            PATH: paths
        }

        # target
        target = {
            DOMAIN_IDENTIFICATION: domain_ids,
            RANKING: ranking_target,
            VERBALIZATION: answer_ids.view(-1)
        }

        # compute output
        output = model(input)

        # compute loss
        loss = criterion(output, target)

        # record loss
        losses[RANKING].update(loss[RANKING].data, args.batch_size)
        losses[DOMAIN_IDENTIFICATION].update(loss[DOMAIN_IDENTIFICATION].data, args.batch_size)
        losses[VERBALIZATION].update(loss[VERBALIZATION].data, args.batch_size)
        losses[MULTITASK].update(loss[MULTITASK].data, args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss[args.task].backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'Epoch {epoch+1} {((i+1)/len(train_loader))*100:.2f}% - Dl: {losses[DOMAIN_IDENTIFICATION].val:.4f} - Rl: {losses[RANKING].val:.4f} - Vl: {losses[VERBALIZATION].val:.4f} - Time: {batch_time.sum:0.2f}s')

def validate(loader, model, criterion):
    losses = {
        RANKING: AverageMeter(),
        DOMAIN_IDENTIFICATION: AverageMeter(),
        VERBALIZATION: AverageMeter(),
        MULTITASK: AverageMeter()
    }

    # switch to evaluate mode
    model.eval()

    for batch in tqdm(loader):
        # extract batch data
        domain_ids = batch[DOMAIN_IDS].to(DEVICE)
        domain_emb = batch[DOMAIN_EMB].to(DEVICE)
        paths = batch[PATH].to(DEVICE)
        ranking_target = batch[RANKING_TARGET].to(DEVICE)
        conversation_ids = batch[CONVERSATION_IDS].to(DEVICE)
        answer_ids = batch[ANSWER_IDS].to(DEVICE)

        # input
        input = {
            CONVERSATION_IDS: conversation_ids,
            ANSWER_IDS: answer_ids,
            DOMAIN_EMB: domain_emb,
            PATH: paths
        }

        # target
        target = {
            DOMAIN_IDENTIFICATION: domain_ids,
            RANKING: ranking_target,
            VERBALIZATION: answer_ids.view(-1)
        }

        # compute output
        output = model(input)

        # compute loss
        loss = criterion(output, target)

        # record loss
        losses[RANKING].update(loss[RANKING].data, args.batch_size)
        losses[DOMAIN_IDENTIFICATION].update(loss[DOMAIN_IDENTIFICATION].data, args.batch_size)
        losses[VERBALIZATION].update(loss[VERBALIZATION].data, args.batch_size)
        losses[MULTITASK].update(loss[MULTITASK].data, args.batch_size)

    return losses

if __name__ == '__main__':
    main()