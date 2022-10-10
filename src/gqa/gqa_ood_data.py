# @File  :gqa_ood_data.py
# @Time  :2021/1/27
# @Desc  :
import os
import json
import h5py

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from param import args
from utils import load_json, load_pickle

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

VG_GQA_IMGFEAT_ROOT = 'data/gqa_imgfeat/'


class GQADataset:
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa_ood/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(
            open("data/gqa_ood/trainval_ans2label.json"))
        self.label2ans = json.load(
            open("data/gqa_ood/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection image object features
        print(f'Loading obj_h5 data from {dataset.splits[0]}')
        self.obj_h5 = h5py.File(os.path.join(
            VG_GQA_IMGFEAT_ROOT, f'{dataset.splits[0]}_obj36.h5'), 'r')
        self.obj_info = load_json(os.path.join(
            VG_GQA_IMGFEAT_ROOT, f'{dataset.splits[0]}_obj36_info.json'))
        self.obj_info = {datum['img_id']: datum
                         for datum in self.obj_info}

        # Loading object attribute-class cosin similarity matrix
        self.adj_h5 = h5py.File(os.path.join(
            VG_GQA_IMGFEAT_ROOT,
            f'{dataset.splits[0]}_obj36_adj_v2.h5'), 'r')

        # Load answer embedding
        # self.ans_embed = load_pickle(os.path.join(
        #     'data/gqa_ood/', 'answer_embeds.pickle'))

        # Only kept the data with loaded image features
        self.data = []
        # if dataset.splits[0] == 'testdev_all':
        #     for datum in self.raw_dataset.data:
        #         if datum['img_id'] in self.obj_info.keys():
        #             self.data.append(datum)
        # else:
        for datum in self.raw_dataset.data:
            for ans, score in datum['label'].items():
                if ans in self.raw_dataset.ans2label \
                        and datum['img_id'] in self.obj_info.keys():
                    self.data.append(datum)

        if args.tiny:
            self.data = self.data[:topk]

        print("Use %d data in torch dataset" % (len(self.data)))
        print("*" * 80)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        # ques_id = datum['question_id']
        # ques = datum['sent']

        # Get image info
        obj_num = self.obj_info[img_id]['num_boxes']
        boxes = self.obj_h5[f'{img_id}']['boxes'][:]
        feats = self.obj_h5[f'{img_id}']['features'][:]
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h = self.obj_info[img_id]['img_h'],
        img_w = self.obj_info[img_id]['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Create target
        if 'label' in datum:
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in datum['label'].items():
                target[self.raw_dataset.ans2label[ans]] = score
            # top_ans = [[k, v] for k, v in sorted(
            #     datum['label'].items(), key=lambda i: i[1], reverse=True)]
            # top_ans_emb = None
            # for tup in top_ans:
            #     if tup[0] in self.ans_embed.keys():
            #         top_ans_emb = F.normalize(torch.tensor(
            #             self.ans_embed[str(tup[0])]).float(), dim=-1)
            #         break
            # if top_ans_emb is None:
            #     top_ans_emb = F.normalize(torch.rand([300]).float(), dim=-1)
            return datum['question_id'], feats, boxes, datum['sent'], target, \
                self.adj_h5[f'{img_id}'][:],  # top_ans_emb  # adj
        else:
            return datum['question_id'], feats, boxes, datum['sent']


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            if ans in datum['label']:
                score += datum['label'][ans]
        return score / len(quesid2ans)

    @staticmethod
    def dump_result(quesid2ans: dict, path):
        """
        Dump the result to a gqa_ood-challenge submittable json file.
        gqa_ood json file submission requirement:
            results = [result]
            result = {
                "questionId": str,
                # Note: it's a actually an int number but
                # the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
