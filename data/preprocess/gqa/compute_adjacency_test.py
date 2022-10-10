# @File  :compute_adjacency.py
# @Time  :2021/1/14
# @Desc  :
import os
import h5py
import numpy as np
from tkinter import _flatten
from tqdm import tqdm

import torch

from src.utils import read_txt

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()


def extract_bert_embedding(sent):
    inputs = tokenizer.encode(sent, return_tensors="pt")
    # sequence_output, pooled_output = model(inputs)
    _, pooled_output = model(inputs.cuda())
    
    return pooled_output


def compute_cosin_sim(matrix):
    adj_cos = torch.zeros((36, 36), dtype=torch.float32).cuda()
    for i in range(36):
        for j in range(36):
            if i != j:
                adj_cos[i, j] = torch.cosine_similarity(
                    matrix[i], matrix[j], dim=0, eps=1e-6)
    return adj_cos


def compute_cosin_sim_v2(matrix1, matrix2):
    adj_cos = torch.zeros((36, 36), dtype=torch.float32).cuda()
    for i in range(36):
        for j in range(36):
            if j >= i:
                adj_cos[i, j] = torch.cosine_similarity(
                    matrix1[i], matrix2[j], dim=0, eps=1e-6)
    return adj_cos + adj_cos.transpose(0, 1)


def count_number(obj_h5):
    obj_ids = [list(set(obj_h5[f'{img_id}']['objects_id'][:].tolist()))
               for img_id in obj_h5.keys()]
    attr_ids = [list(set(obj_h5[f'{img_id}']['attrs_id'][:].tolist()))
                for img_id in obj_h5.keys()]
    
    obj_ids = list(set(list(_flatten(obj_ids))))
    attr_ids = list(set(list(_flatten(attr_ids))))
    
    print(f'obj: len: {len(obj_ids)}, max: {max(obj_ids)}')
    print(f'attr: len: {len(attr_ids)}, max: {max(attr_ids)}')


def main():
    # torch.set_num_threads(8)
    mode = 'testdev_all'
    obj_h5 = h5py.File(os.path.join(
        'data/gqa_imgfeat/', f'{mode}_obj36.h5'), 'r')
    obj_label = read_txt(f'data/gqa_imgfeat/objects_vocab.txt')
    attr_label = read_txt(f'data/gqa_imgfeat/attributes_vocab.txt')
    id2obj = {i: obj_label[i] for i in range(len(obj_label))}
    id2attr = {i: attr_label[i] for i in range(len(attr_label))}
    
    h5_adj = h5py.File(os.path.join(
        'data/gqa_imgfeat/', f'{mode}_obj36_adj_v2.h5'), 'w')
    tbar = tqdm(total=len(obj_h5.keys()), ncols=80)
    for img_id in obj_h5.keys():
        obj_ids = obj_h5[f'{img_id}']['objects_id'][:].tolist()
        attr_ids = obj_h5[f'{img_id}']['attrs_id'][:].tolist()
        matrix_class = []
        matrix_attribute = []
        for o, a in zip(obj_ids, attr_ids):
            sent_class = f'{id2obj[o]}'
            sent_attribute = f'{id2attr[a]}'
            matrix_class.append(extract_bert_embedding(sent_class))
            matrix_attribute.append(extract_bert_embedding(sent_attribute))
        matrix_class = torch.cat(matrix_class, dim=0)
        matrix_attribute = torch.cat(matrix_attribute, dim=0)
        # matrix = matrix_class + matrix_attribute
        # matrix = compute_cosin_sim(matrix)
        matrix = compute_cosin_sim_v2(matrix_class, matrix_attribute)
        matrix = matrix/matrix.max()
        h5_adj.create_dataset(name=f'{img_id}',
                              data=matrix.detach().cpu().numpy(),
                              dtype=np.float32)
        tbar.update(1)
    tbar.close()


if __name__ == '__main__':
    main()
