# X-GGM 

PyTorch code of our paper: [X-GGM: Graph Generative Modeling for Out-of-Distribution Generalization in Visual Question Answering](https://dl.acm.org/doi/abs/10.1145/3474085.3475350) (MM 2021),

This implementation is based on [LXMERT](https://github.com/airsplay/lxmert). Thanks for their pioneering work.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Please see data/README.md for details and download from [here](https://drive.google.com/file/d/1mFLXIHmjMmCVmAJaoj9gbV96buRmp44D/view?usp=sharing).

```angular2html
├── data
│   ├── gqa_imgfeat
│   │   ├── testdev_all_obj36_adj_v2.h5
│   │   ├── testdev_all_obj36.h5
│   │   ├── testdev_all_obj36_info.json
│   │   ├── testdev_head_obj36.h5
│   │   ├── testdev_head_obj36_info.json
│   │   ├── testdev_tail_obj36.h5
│   │   ├── testdev_tail_obj36_info.json
│   │   ├── train_obj36_adj_v2.h5
│   │   ├── train_obj36.h5
│   │   ├── train_obj36_info.json
│   │   ├── val_all_obj36_adj_v2.h5
│   │   ├── val_all_obj36.h5
│   │   ├── val_all_obj36_info.json
│   │   ├── val_tail_obj36_adj_v2.h5
│   │   ├── val_tail_obj36.h5
│   │   └── val_tail_obj36_info.json
│   ├── gqa_ood
│   │   ├── answer_embeds.pickle
│   │   ├── testdev_all.json
│   │   ├── testdev_head.json
│   │   ├── testdev_tail.json
│   │   ├── train.json
│   │   ├── trainval_ans2label.json
│   │   ├── trainval_label2ans.json
│   │   ├── val_all.json
│   │   ├── val_head.json
│   │   └── val_tail.json
│   ├── lxmert
│   │   └── all_ans.json
│   ├── mscoco_imgfeat
│   │   ├── dev_test_obj36_adj_v2.h5
│   │   ├── dev_test_obj36.h5
│   │   ├── dev_test_obj36_info.json
│   │   ├── test_obj36_adj_v2.h5
│   │   ├── test_obj36.h5
│   │   ├── test_obj36_info.json
│   │   ├── train_obj36_adj_v2.h5
│   │   ├── train_obj36.h5
│   │   ├── train_obj36_info.json
│   │   ├── val_obj36_adj.h5
│   │   ├── val_obj36.h5
│   │   └── val_obj36_info.json
│   ├── vqa
│   │   ├── minival.json
│   │   ├── nominival.json
│   │   ├── train.json
│   │   ├── trainval_ans2label.json
│   │   └── trainval_label2ans.json
│   └── vqacpv2
│       ├── dev_test_annotations.json
│       ├── test_annotations.json
│       ├── train_annotations.json
│       ├── trainval_ans2label.json
│       ├── trainval_label2ans.json
│       └── val_annotations.json
```

## Training and Testing

Please see the parameter settings and running details in the scripts.

- For VQA-CP v2

```bash
bash script/vqacpv2.sh
```

- For GQA-OOD 

```bash
bash script/gqa_ood.sh
```

## Citation

If you find our code is helpful for your research, please cite:
```
@inproceedings{jiang2021x,
  title={X-GGM: Graph generative modeling for out-of-distribution generalization in visual question answering},
  author={Jiang, Jingjing and Liu, Ziyi and Liu, Yifan and Nan, Zhixiong and Zheng, Nanning},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={199--208},
  year={2021}
}
```

