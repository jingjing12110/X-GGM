# @File  :tsv2h5.py
# @Time  :2020/12/30
# @Desc  :
import os
import sys
import glob

import h5py
import csv
import base64
import numpy as np
from tqdm import tqdm

from src.utils import load_json, save_json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def combine_tsv():
    tsv_list = glob.glob('./data/gqa_imgfeat/*.tsv')
    
    for i in tsv_list:
        fr = open(i, 'rb').read()
        with open(os.path.join('./data/gqa_imgfeat/',
                               'trainvaltest_obj36.tsv'), 'ab') as f:
            f.write(fr)


def filter_tsv(tsv_file, img_ids):
    h5_data = h5py.File('data/gqa_imgfeat/testdev_tail_obj36.h5', "w")
    train_info = []
    pbar = tqdm(total=len(img_ids))
    
    with open(tsv_file) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            img_id = item['img_id']
            if img_id in img_ids:
                h5_temp = h5_data.create_group(f'{img_id}')
                info_temp = {'img_id': img_id}
                for key in ['img_h', 'img_w', 'num_boxes']:
                    info_temp[key] = int(item[key])
                train_info.append(info_temp)
                boxes = int(item['num_boxes'])
                decode_config = [
                    ('objects_id', (boxes,), np.int64),
                    ('objects_conf', (boxes,), np.float32),
                    ('attrs_id', (boxes,), np.int64),
                    ('attrs_conf', (boxes,), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]),
                                              dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    
                    h5_temp.create_dataset(name=key, data=item[key], dtype=dtype)
                
                pbar.update(1)
        print(f'i={i}')
    save_json(train_info, 'data/gqa_imgfeat/testdev_tail_obj36_info.json')
    pbar.close()


def main():
    tsv_file = 'data/gqa_imgfeat/gqa_testdev_obj36.tsv'
    train_file = os.path.join('./data/gqa_ood/', 'testdev_tail.json')
    targets = load_json(train_file)
    
    img_ids = [t['img_id'] for t in targets]
    img_ids = list(set(img_ids))
    filter_tsv(tsv_file, img_ids)


if __name__ == '__main__':
    main()
