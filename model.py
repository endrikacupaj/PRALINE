import h5py
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

# import constants
from constants import *

# PRALINE
class PRALINE(nn.Module):
    def __init__(self, domains):
        super(PRALINE, self).__init__()
        self.domain_pointer         = DomainIdentificationPointer(domains)
        self.ranking_module         = RankingModule()
        self.verbalization_module   = VerbalizationModule()

    def forward(self, input):
        verbalization_out = self.verbalization_module(input[CONVERSATION_IDS], input[ANSWER_IDS])
        domain_pointer_out = self.domain_pointer(verbalization_out[ENCODER_OUT].max(dim=1).values.unsqueeze(1))
        ranking_out = self.ranking_module(verbalization_out[ENCODER_OUT].max(dim=1).values, input[DOMAIN_EMB], input[PATH])

        return {
            DOMAIN_IDENTIFICATION: domain_pointer_out,
            RANKING: ranking_out,
            VERBALIZATION: verbalization_out[LOGITS]
        }

class LearnSequence(nn.Module):
    def __init__(self, in_dim=args.in_dim, emb_dim=args.emb_dim):
        super(LearnSequence, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.embedding(x.unsqueeze(1))

class DomainIdentificationPointer(nn.Module):
    def __init__(self, domains):
        super(DomainIdentificationPointer, self).__init__()
        # domain embeddings
        self.domains = domains
        self.damain_embed = []
        with h5py.File(f'{ROOT_PATH}/{args.data_path}/domains.h5', 'r') as domain_h5:
            for domain in self.domains:
                self.damain_embed.append(torch.from_numpy(domain_h5[domain][()]).float())
        self.damain_embed = torch.stack(self.damain_embed).to(DEVICE)

        # pointer network
        self.linear_in = nn.Linear(args.in_dim, args.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.tahn = nn.Tanh()
        self.linear_out = nn.Linear(args.emb_dim, 1)

    def forward(self, x):
        embed = self.linear_in(self.damain_embed).unsqueeze(0)
        x = x.expand(x.shape[0], embed.shape[1], x.shape[-1])
        x = x + embed.expand(x.shape[0], embed.shape[1], embed.shape[-1])
        x = self.tahn(x)
        x = self.linear_out(x)

        return x.squeeze(-1)

class RankingModule(nn.Module):
    def __init__(self):
        super(RankingModule, self).__init__()

        self.learn_conv_domain = LearnSequence(in_dim=2*args.in_dim)
        self.learn_path        = LearnSequence()

    def forward(self, conversation, domain, path):
        return {
            CONVERSATION: self.learn_conv_domain(torch.cat([conversation, domain], dim=-1)).squeeze(1),
            PATH: self.learn_path(path).squeeze(1)
        }

class VerbalizationModule(nn.Module):
    def __init__(self):
        super(VerbalizationModule, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(BART_MODEL)

    def forward(self, conversation_ids, answer_ids):
        output = self.bart(input_ids=conversation_ids, labels=answer_ids)
        return {
            LOGITS: output.logits.view(-1, output.logits.shape[-1]),
            ENCODER_OUT: output.encoder_last_hidden_state
        }

    def predict(self, input_ids):
        self.eval()
        return self.bart.generate(input_ids).squeeze(0)
