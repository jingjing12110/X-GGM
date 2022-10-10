# @File  :gqa_ood.py
# @Time  :2021/1/27
# @Desc  :
import os
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from module.graph_utils import add_feature_noise_v2 as add_feature_noise
from module.graph_utils import add_edge_noise_v2 as add_edge_noise
from gqa.gqa_ood_model import GQAModel
from gqa.gqa_ood_data import GQADataset, GQATorchDataset, GQAEvaluator

if args.tiny:
    from torch.utils.data.dataloader import DataLoader
else:
    from tools.data_loader import DataLoaderX as DataLoader

TIMESTAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False,
                   drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def loss_func(score, grad_log_q_noise, sigma=0.2):
    cur_loss = 0.5 * sigma ** 2 * ((score - grad_log_q_noise) ** 2).sum(
        dim=[-1, -2]).mean()
    return cur_loss / (score.shape[-1] * score.shape[-2])


def compute_kl_loss(x, y):
    px = F.softmax(x, dim=-1)
    log_px = F.log_softmax(x, dim=-1)
    py = F.softmax(y, dim=-1)
    log_py = F.log_softmax(y, dim=-1)
    kl_loss = F.kl_div(log_px, py, reduction='none') + F.kl_div(
        log_py, px, reduction='none')
    return kl_loss.mean()


def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.mean(F.relu(1. - dis_real))
    loss += torch.mean(F.relu(1. + dis_fake))
    return loss


class GQA:
    def __init__(self):
        self.train_tuple = get_data_tuple(
            args.train,
            bs=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        if args.valid != "":
            # valid_bsize = 10 * args.batch_size if args.multiGPU else 512
            self.valid_tuple = get_data_tuple(
                args.valid,
                bs=2 * args.batch_size,
                shuffle=False,
                drop_last=False
            )
        else:
            self.valid_tuple = None

        # self.ans_embed = load_pickle(os.path.join(
        #     'data/gqa_ood/', 'answer_embeds.pickle'))
        # self.ans_embed = [e for _, e in self.ans_embed.items()]
        # self.ans_embed = F.normalize(
        #     torch.tensor(self.ans_embed).float(), dim=-1)

        # Model
        self.model = GQAModel(
            self.train_tuple.dataset.num_answers,
            gnn=args.gnn,  # 'GCN',
            n_layers=args.num_layer
        )

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.rec_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            base_param = list(map(id, self.model.lxrt_encoder.parameters()))
            down_task_params = filter(
                lambda p: id(p) not in base_param, self.model.parameters())
            parameters = [{'params': down_task_params},
                          {'params': self.model.lxrt_encoder.parameters(),
                           'lr': args.lr}
                          ]
            self.optim = BertAdam(parameters,
                                  lr=4 * args.lr,
                                  warmup=0.1,
                                  t_total=2 * t_total)
            # self.optim = BertAdam(list(self.model.parameters()),
            #                       lr=args.lr,
            #                       warmup=0.1,
            #                       t_total=2*t_total)
        else:
            self.optim = args.optimizer(
                list(self.model.parameters()),
                args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = None
        if args.test is None and args.tf_writer:
            self.writer = SummaryWriter(os.path.join(
                self.output, f'{TIMESTAMP}/logs'))

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (
            lambda x: tqdm(x, total=len(loader), ncols=80)) if args.tqdm else (
            lambda x: x)

        val_in_epoch = np.linspace(0, len(loader), 5, dtype=np.int)[1:-1]  # 3
        best_valid, total_loss, train_iter = 0., 0., 0
        d_loss_1, d_loss_2 = 0., 0.
        loss_sm_1, loss_sm_2 = 0., 0.
        loss_grad_1, loss_grad_2 = 0., 0.
        for epoch in range(args.epochs):
            quesid2ans = {}

            for i, (ques_id, feats, boxes, sent, target, adj_true, # a_embed_true
                    ) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.model.zero_grad()
                self.optim.zero_grad()

                target = target.cuda(non_blocking=True)
                feat_seq, _, x = self.model(
                    feats.cuda(), boxes.cuda(), sent)
                # GGM
                # remove triu
                adj_true = adj_true.cuda()
                adj_true = adj_true.triu(1) + adj_true.tril(-1)

                random_num = random.randint(1, 10)
                if random_num <= args.delta:
                    # relation generation
                    adj_noise = torch.zeros_like(adj_true)
                    adj_temp = torch.ones_like(adj_true).triu(1)
                    adj_noise[adj_temp == 1] = self.model.encoder_adj(
                        x).view(-1)
                    adj_noise = adj_noise + adj_noise.transpose(1, 2)

                    adj_noise, grad_log_noise = add_edge_noise(
                        adj_noise, sigma=args.sigma)  # add gauss noise
                    node_feats, adj_noise = self.model.generator(
                        feat_seq[1], adj_noise)
                    # Adversarial loss
                    d_loss = compute_kl_loss(
                        adj_noise, adj_true) * target.size(1)
                    loss_grad = loss_func(
                        adj_noise, grad_log_noise, sigma=args.sigma)
                    loss_sm = 12 * d_loss + loss_grad
                    # loss_sm = loss_grad

                    # VQA head
                    x_gen = self.model.fusion_fc(
                        torch.cat([x, torch.tanh(node_feats.mean(1))], dim=-1)
                    )
                    # gold_embeds = self.model.ans_embed_proj(
                    #     a_embed_true.cuda())
                    # # TODO: modify self.model.gen_ans_embed
                    # gen_embeds = F.normalize(self.model.gen_ans_embed(
                    #     x_gen), dim=-1)
                    #
                    # positive_dist = F.cosine_similarity(
                    #     gen_embeds, gold_embeds, dim=1)
                    # all_a_embeds = self.model.ans_embed_proj(
                    #     self.ans_embed.cuda())
                    # all_a_embeds = torch.stack(
                    #     [all_a_embeds] * target.shape[0])
                    # gen_embeds = torch.cat(
                    #     [gen_embeds.unsqueeze(1)] * all_a_embeds.shape[1], 1)
                    # d_logit = F.cosine_similarity(
                    #     gen_embeds, all_a_embeds, dim=2)
                    # loss_nce = -1 * torch.log(torch.exp(
                    #     positive_dist
                    # ).squeeze(-1) / torch.exp(d_logit).sum(-1))
                    # loss_nce = loss_nce.mean()

                    logit = self.model.logit_fc(x_gen)
                    loss = self.bce_loss(logit, target) * logit.size(1)
                    # loss += loss_nce + 8 * loss_sm
                    loss += 6 * loss_sm
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optim.step()
                else:
                    # **generating representation**
                    # node_feats = torch.zeros_like(feat_seq[1])
                    node_feats = x.unsqueeze(1).repeat(1, 36, 1)
                    node_feats = self.model.node_fc(node_feats)
                    node_feats, feat_grad = add_feature_noise(
                        node_feats, sigma=args.sigma)

                    node_feats, _ = self.model.generator(node_feats, adj_true)
                    # Adversarial loss
                    d_loss = compute_kl_loss(
                        node_feats, feat_seq[1]) * target.size(1)
                    loss_grad = loss_func(
                        node_feats, feat_grad, sigma=args.sigma)
                    loss_sm = 0.15 * d_loss + 6 * loss_grad
                    # loss_sm = 6 * loss_grad

                    # VQA head
                    x_gen = self.model.fusion_fc(
                        torch.cat([x, torch.tanh(node_feats.mean(1))], dim=-1)
                    )
                    # gold_embeds = self.model.ans_embed_proj(
                    #     a_embed_true.cuda())
                    # # TODO
                    # gen_embeds = F.normalize(self.model.gen_ans_embed(
                    #     x_gen), dim=-1)
                    #
                    # positive_dist = F.cosine_similarity(
                    #     gen_embeds, gold_embeds, dim=1)
                    # all_a_embeds = self.model.ans_embed_proj(
                    #     self.ans_embed.cuda())
                    # all_a_embeds = torch.stack(
                    #     [all_a_embeds] * target.shape[0])
                    # gen_embeds = torch.cat(
                    #     [gen_embeds.unsqueeze(1)] * all_a_embeds.shape[1], 1)
                    # d_logit = F.cosine_similarity(
                    #     gen_embeds, all_a_embeds, dim=2)
                    # loss_nce = -1 * torch.log(torch.exp(
                    #     positive_dist
                    # ).squeeze(-1) / torch.exp(d_logit).sum(-1))
                    # loss_nce = loss_nce.mean()

                    logit = self.model.logit_fc(x_gen)
                    loss = self.bce_loss(logit, target) * target.size(1)
                    # loss += loss_nce + 1.1 * loss_sm
                    loss += 1.1 * loss_sm
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    self.optim.step()

                self.optim.zero_grad()
                self.model.zero_grad()
                _, _, x = self.model(feats.cuda(), boxes.cuda(), sent)
                logit = self.model.logit_fc(x)
                loss = self.bce_loss(logit, target) * target.size(1)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                total_loss += loss.detach() / logit.size(1)
                for qid, l in zip(ques_id, logit.max(1)[1].cpu().numpy()):
                    quesid2ans[qid] = dset.label2ans[l]

                if args.tf_writer:
                    self.writer.add_scalar(
                        'Train/batch_loss',
                        loss.detach(),
                        train_iter)
                    self.writer.add_scalar(
                        'Train/average_loss',
                        total_loss / (train_iter + 1),
                        train_iter)
                    self.writer.add_scalar(
                        'lr',
                        self.optim.state_dict()['param_groups'][0]['lr'],
                        train_iter)
                    # self.writer.add_scalar(
                    #     'lr2',
                    #     self.optim.state_dict()['param_groups'][1]['lr'],
                    #     train_iter)
                    self.writer.add_scalar(
                        'Train/node_gen/all_loss',
                        loss_sm_1,
                        train_iter)
                    self.writer.add_scalar(
                        'Train/node_gen/dis_loss',
                        d_loss_1,
                        train_iter)
                    self.writer.add_scalar(
                        'Train/node_gen/grad_loss',
                        loss_grad_1,
                        train_iter)
                    self.writer.add_scalar(
                        'Train/repr_gen/all_loss',
                        loss_sm_2,
                        train_iter)
                    self.writer.add_scalar(
                        'Train/repr_gen/dis_loss',
                        d_loss_2,
                        train_iter)
                    self.writer.add_scalar(
                        'Train/repr_gen/grad_loss',
                        loss_grad_2,
                        train_iter)
                    train_iter += 1

                # epoch ??validation
                if i in val_in_epoch:
                    valid_score = self.evaluate(eval_tuple)
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")
                        # self.save(f"BEST_{best_valid * 100:.2f}")
                    if args.tf_writer:
                        self.writer.add_scalar(
                            'Val/acc', valid_score, epoch)
                    print(f"\nEpoch {epoch}: Iter {i}: "
                          f"Valid {valid_score * 100.:.2f}\n"
                          f"Epoch {epoch}: Iter {i}: "
                          f"Best {best_valid * 100.:.2f}\n")
            # *compute train score*
            train_score = evaluator.evaluate(quesid2ans) * 100.
            log_str = f"\nEpoch {epoch}: " \
                      f"Train {train_score:.2f}\n"
            if args.tf_writer:
                self.writer.add_scalar(
                    'Train/acc', train_score, epoch)
            # Do validation for each epoch
            if self.valid_tuple is not None:
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                self.save(f"BEST_{epoch}")
                if args.tf_writer:
                    self.writer.add_scalar(
                        'Val/acc', valid_score, epoch)
                log_str += f"Epoch {epoch}: Valid {valid_score * 100.:.2f}\n" + \
                           f"Epoch {epoch}: Best {best_valid * 100.:.2f}\n"
            print(log_str, end='')
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        # self.save("LAST")
        if args.tf_writer:
            self.writer.close()

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        tbar = tqdm(total=len(loader), ascii=True, desc='Test', ncols=80)
        for i, datum_tuple in enumerate(loader):
            # Avoid seeing ground truth
            ques_id, feats, boxes, sent = datum_tuple[:4]
            with torch.no_grad():
                _, _, x = self.model(feats.cuda(), boxes.cuda(), sent)
                logit = self.model.logit_fc(x)
                for qid, l in zip(ques_id, logit.max(1)[1].cpu().numpy()):
                    quesid2ans[qid] = dset.label2ans[l]
            tbar.update(1)
        tbar.close()
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (datum_tuple) in enumerate(loader):
            ques_id, target = datum_tuple[0], datum_tuple[4]
            # _, label = target.max(1)
            for qid, l in zip(ques_id, target.max(1)[1].cpu().numpy()):
                quesid2ans[qid] = dset.label2ans[l]  # ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    from src.utils import save_json

    print(args)
    # Build Class
    gqa = GQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        print("*" * 80)
        # Always loading all data in test
        args.fast = args.tiny = False
        if 'test' in args.test:
            gqa.predict(
                get_data_tuple(args.test, bs=512,
                               shuffle=False, drop_last=False),
                dump=os.path.join(
                    args.output,
                    f"{args.test}_predict_{args.load.split('/')[-1]}.json"
                )
            )
        elif 'val' in args.test:
            result = gqa.evaluate(
                get_data_tuple(args.test, bs=512,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, f'{args.test}_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        save_json(vars(args), os.path.join(args.output, 'args.json'))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            # print(
            #     "Valid Oracle: %0.2f" % (
            #     gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)

