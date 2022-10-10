# coding=utf-8
# Copyleft 2019 Project LXRT
import os
import sys
import csv
import base64
import h5py
import pickle

import time
import json
import errno

import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
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
                item[key].setflags(write=False)
            
            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (
        len(data), fname, elapsed_time))
    
    return data


def load_obj_h5(data_root, mode, topk=None):
    start_time = time.time()
    fname = os.path.join(data_root, f'{mode}_obj36.h5')
    finfo = os.path.join(data_root, f'{mode}_obj36_info.json')
    data_info = load_json(finfo)
    data_info_dict = {datum['img_id']: datum
                      for datum in data_info}
    print(f"Start to load Faster-RCNN detected objects from {fname}")
    data = []
    h5_file = h5py.File(fname, 'r')
    for key in h5_file.keys():
        temp = {'img_id': int(key)}
        for k in ['img_h', 'img_w', 'num_boxes']:
            temp[k] = data_info_dict[int(key)][k]
        for k in ['attrs_conf', 'attrs_id', 'boxes',
                  'features', 'objects_conf', 'objects_id']:
            temp[k] = h5_file[key].get(k)[:]
        data.append(temp)
        if topk is not None and len(data) == topk:
            break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (
        len(data), fname, elapsed_time))
    
    return data

    
def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
    
def save_pickle(data, data_path, highest=False):
    protocol = 2 if highest else 0
    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def read_txt(file):
    with open(file, "r") as f:
        data = [line.strip('\n') for line in f.readlines()]
    return data


def write_txt(file, s):
    with open(file, 'a+') as f:
        f.write(s)


if __name__ == '__main__':
    data_file = 'data/mscoco_imgfeat/val_obj36.h5'
    load_obj_h5(data_file, 5000)
