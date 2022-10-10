# @File  :vqacpv2_data.py
# @Time  :2021/1/28
# @Desc  :
import json
import os
import h5py

import numpy as np
import torch
from torch.utils.data.dataloader import Dataset

from param import args
from utils import load_obj_tsv, save_json, load_json, load_obj_h5

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqacpv2/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'


class VQADataset:
    def __init__(self, splits: str, space=1):
        self.name = splits
        self.splits = splits.split(',')
        
        # Convert list to dict (for evaluation)
        self.data = load_json(
            f'data/vqacpv2/{self.name}_annotations.json')
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        print(f"Loading {self.name} data: {len(self.data)}")
        
        # Answers
        self.ans2label = load_json(
            f'data/vqacpv2/trainval_ans2label.json')
        self.label2ans = load_json(
            f'data/vqacpv2/trainval_label2ans.json')
        assert len(self.ans2label) == len(self.label2ans)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __len__(self):
        return len(self.data)


class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset
        
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None
        
        # Loading detection features to img_data
        print(f'Loading obj_h5 data from {dataset.splits[0]}')
        self.obj_h5 = h5py.File(os.path.join(
            MSCOCO_IMGFEAT_ROOT, f'{dataset.splits[0]}_obj36.h5'), 'r')
        self.obj_info = load_json(os.path.join(
            MSCOCO_IMGFEAT_ROOT, f'{dataset.splits[0]}_obj36_info.json'))
        self.obj_info = {datum['img_id']: datum
                         for datum in self.obj_info}
        
        # Loading object attribute-class cosin similarity matrix
        if dataset.splits[0] in ['train', 'dev_test']:
            self.adj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT,
                f'{dataset.splits[0]}_obj36_adj_v2.h5'), 'r')
        
        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['image_id'] in self.obj_info.keys():
                self.data.append(datum)
        if args.tiny:
            self.data = self.data[:topk]
        print("Use %d data in torch dataset" % (len(self.data)))
        print("*" * 80)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        # img_id = datum['img_id']
        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']
        
        # Get image info
        img_info = self.obj_h5[f'{img_id}']
        obj_num = self.obj_info[img_id]['num_boxes']
        feats = img_info['features'][:]
        boxes = img_info['boxes'][:]
        assert obj_num == len(boxes) == len(feats)
        
        # Normalize the boxes (to 0 ~ 1)
        img_h = self.obj_info[img_id]['img_h'],
        img_w = self.obj_info[img_id]['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)
        
        # Provide label (target)
        if 'label' in datum:
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in zip(datum['label'], datum['score']):
                target[ans] = score
            return ques_id, feats, boxes, ques, target, \
                self.adj_h5[f'{img_id}'][:]
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset
    
    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = dict(zip(datum['label'], datum['score']))
            aid = self.dataset.ans2label[ans]
            if aid in label:
                score += label[aid]
        return score / len(quesid2ans)
    
    @staticmethod
    def dump_result(quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online
         evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    dset = VQADataset(splits='val')
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)

