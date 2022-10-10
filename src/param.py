import argparse
import random
import ast

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        print("Optimizer: Using AdamW")
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim
    
    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='val')
    parser.add_argument("--test", default=None)
    
    # Training Hyper-parameters
    parser.add_argument('--bs', dest='batch_size', type=int,
                        default=8)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')
    parser.add_argument('--fp16', action='store_const', default=False, const=True)
    parser.add_argument('--space', type=int, default=1, choices=[1, 9, 12])
    parser.add_argument('--tf_writer', default=True, type=ast.literal_eval)
    
    # Debugging
    parser.add_argument('--output', type=str, default='snap/debug')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)
    
    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str,
                        default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str,
                        default='snap/pretrained/model',
                        help='Load the pre-trained LXMERT model with QA answer '
                             'head.')
    parser.add_argument("--fromScratch", dest='from_scratch',
                        action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA'
                             'is set the model would be trained from scratch. '
                             'If --fromScratch is not specified, the model would '
                             'load BERT-pre-trained weights by default. ')
    
    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const',
                        default=False, const=True)
    
    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int,
                        help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int,
                        help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int,
                        help='Number of object Relationship layers.')
    
    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const',
                        default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict',
                        action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const',
                        default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses',
                        default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15,
                        type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15,
                        type=float)
    
    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False,
                        const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0, type=int)
    
    # OOD Generalization config
    parser.add_argument('--eg', dest='edge_gnn', default=None)
    parser.add_argument("--tmode", default='OOD', type=str,
                        help="['OOD', 'ID']")
    parser.add_argument('--gnn', default='GCN', type=str)
    parser.add_argument('--num_layer', default=2, type=int)
    parser.add_argument('--sigma', default=1.0, type=float,
                        help='gaussian noise')
    parser.add_argument('--delta', default=5, type=int)

    # Parse the arguments.
    args = parser.parse_args()
    
    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    return args


args = parse_args()
