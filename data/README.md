# Data Processing

## VQA-CP v2

- Downloading image features from LXMERT [https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/](https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/).
- Converting .tsv to .h5 (preprocess/vqa/tsv2h5.py)

```angular2html
├── mscoco_imgfeat
│   ├── dev_test_obj36.h5
│   ├── dev_test_obj36_info.json
│   ├── test_obj36.h5
│   ├── test_obj36_info.json
│   ├── train_obj36.h5
│   ├── train_obj36_info.json
│   ├── val_obj36.h5
│   └── val_obj36_info.json
```

- Downloading adjacency matrix from [here](https://drive.google.com/file/d/1mFLXIHmjMmCVmAJaoj9gbV96buRmp44D/view?usp=sharing)

## GQA-OOD

- Downloading image features from LXMERT [https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/](https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/).
- Converting .tsv to .h5 (preprocess/gqa/tsv2h5.py)

```angular2html
├── gqa_imgfeat
│   ├── testdev_all_obj36.h5
│   ├── testdev_all_obj36_info.json
│   ├── testdev_head_obj36.h5
│   ├── testdev_head_obj36_info.json
│   ├── testdev_tail_obj36.h5
│   ├── testdev_tail_obj36_info.json
│   ├── train_obj36.h5
│   ├── train_obj36_info.json
│   ├── val_all_obj36.h5
│   ├── val_all_obj36_info.json
│   ├── val_tail_obj36.h5
│   └── val_tail_obj36_info.json
```

- Downloading adjacency matrix from [here](https://drive.google.com/file/d/1mFLXIHmjMmCVmAJaoj9gbV96buRmp44D/view?usp=sharing)




