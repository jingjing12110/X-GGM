# @File  :tsv2h5.py
# @Time  :2020/12/28
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
    tsv_list = glob.glob('data/mscoco_imgfeat/*.tsv')
    
    for i in tsv_list:
        fr = open(i, 'rb').read()
        with open(os.path.join('data/mscoco_imgfeat/',
                               'trainval2014_obj36.tsv'), 'ab') as f:
            f.write(fr)


def filter_tsv(tsv_file, img_ids):
    h5_data = h5py.File('data/mscoco_imgfeat/dev_test_obj36.h5', "w")
    train_info = []
    pbar = tqdm(total=len(img_ids))

    with open(tsv_file) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            img_id = int(item['img_id'].split('_')[-1])
            if img_id in img_ids:
                h5_temp = h5_data.create_group(f'{img_id}')
                info_temp = {'img_id': img_id}
                for key in ['img_h', 'img_w', 'num_boxes']:
                    # item[key] = int(item[key])
                    info_temp[key] = int(item[key])
                    # h5_temp.create_dataset(
                    #     name=key, data=int(item[key]), dtype='i')
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
                    # item[key].setflags(write=False)
                    h5_temp.create_dataset(name=key, data=item[key], dtype=dtype)
                
                # data.append(item)
                pbar.update(1)
    save_json(train_info, 'data/mscoco_imgfeat/dev_test_obj36_info.json')
    pbar.close()
    # return data


def main():
    # tsv_file = 'data/mscoco_imgfeat/trainval2014_obj36.tsv'
    tsv_file = '/media/kaka/SSD1T/DataSet/vqacp2/trainval2014_obj36.tsv'
    train_file = os.path.join(
        'data/vqacpv2/raw_anns/vqacp_v2_dev_testsplit_annotations.json')
    targets = load_json(train_file)
    
    print(f'len target: {len(targets)}')
    img_ids = [t['image_id'] for t in targets]
    img_ids = list(set(img_ids))
    print(f'len img_ids: {len(img_ids)}')
    filter_tsv(tsv_file, img_ids)


if __name__ == '__main__':
    main()
