# @File  :vqacpv2_model.py
# @Time  :2021/2/2
# @Desc  :
import torch
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder, LXRTEncoderFeature
from lxrt.modeling import BertLayerNorm, GeLU

from module.graph_generative_modeling import GINGenerator
from module.graph_generative_modeling import GCNGenerator
from module.graph_generative_modeling import GATGenerator
from module.graph_generative_modeling import Discriminator

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAPlainModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
    
    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        return logit


class VQAModel(nn.Module):
    def __init__(self, num_answers, gnn='GCN', n_layers=2):
        super().__init__()
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoderFeature(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
        if gnn == 'GCN':
            self.generator = GCNGenerator(
                hidden_dim=hid_dim, n_layers=n_layers)
            # self.encoder = GCNPlainEncoder(
            #     hidden_dim=hid_dim, n_layers=n_layers)
        elif gnn == 'GIN':
            self.generator = GINGenerator(
                hidden_dim=hid_dim, n_layers=n_layers)
            # self.encoder = GinPlainEncoder(
            #     hidden_dim=hid_dim, n_layers=n_layers)
        elif gnn == 'GAT':
            self.generator = GATGenerator(
                hidden_dim=hid_dim, n_layers=n_layers)
        else:
            raise ModuleNotFoundError
        
        # self.discriminator_node = Discriminator(hidden_dim=36 * hid_dim)
        # self.discriminator_edge = Discriminator(hidden_dim=36 * 36)
        
        # for relation/node initialization
        self.encoder_adj = nn.Sequential(
            nn.Linear(768, 630),
            nn.Sigmoid()
        )
        self.node_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            GeLU(),
            nn.LayerNorm(hid_dim)
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            GeLU(),
            nn.LayerNorm(hid_dim)
        )
        
        # answer embedding using for NCE loss
        # self.ans_embed_proj = nn.Sequential(
        #     nn.Linear(300, hid_dim),
        #     GeLU(),
        #     BertLayerNorm(hid_dim, eps=1e-12),
        #     nn.Linear(hid_dim, 300)
        # )
        #
        # self.gen_ans_embed = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim * 2),
        #     GeLU(),
        #     BertLayerNorm(hid_dim * 2, eps=1e-12),
        #     nn.Linear(hid_dim * 2, 300)
        # )
    
    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The logit of each answers.
        """
        feat_seq, que_mask, x = self.lxrt_encoder(sent, (feat, pos))
        return feat_seq, que_mask, x


class GQAModelGIN(nn.Module):
    def __init__(self, num_answers, n_layers=2):
        super().__init__()
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoderFeature(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
        self.generator = GINGenerator(hidden_dim=hid_dim, n_layers=n_layers)
        # self.generator_node = NodeGenerator(hidden_dim=hid_dim, n_layers=2)
        # self.generator_edge = EdgeGenerator(hidden_dim=hid_dim, n_layers=2)
        
        self.discriminator_node = Discriminator(hidden_dim=36 * hid_dim)
        self.discriminator_edge = Discriminator(hidden_dim=36 * 36)
        self.encoder_adj = nn.Sequential(
            nn.Linear(768, 630),
            nn.Sigmoid()
        )
    
    def forward(self, feat, pos, sent):
        feat_seq, que_mask, x = self.lxrt_encoder(sent, (feat, pos))
        return feat_seq, que_mask, x


class GQAModelGCN(nn.Module):
    def __init__(self, num_answers, n_layers=2):
        super().__init__()
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoderFeature(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
        self.generator = GCNGenerator(hidden_dim=hid_dim, n_layers=n_layers)
        
        self.discriminator_node = Discriminator(hidden_dim=36 * hid_dim)
        self.discriminator_edge = Discriminator(hidden_dim=36 * 36)
        self.encoder_adj = nn.Sequential(
            nn.Linear(768, 630),
            nn.Sigmoid()
        )
        
        # self.fusion_fc = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.LayerNorm(hid_dim),
        #     nn.Tanh()
        # )
    
    def forward(self, feat, pos, sent):
        feat_seq, que_mask, x = self.lxrt_encoder(sent, (feat, pos))
        return feat_seq, que_mask, x


class GQAModelGAT(nn.Module):
    def __init__(self, num_answers, n_layers=2):
        super().__init__()
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoderFeature(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        
        self.generator = GATGenerator(hidden_dim=hid_dim, n_layers=n_layers)
        
        self.discriminator_node = Discriminator(hidden_dim=36 * hid_dim)
        self.discriminator_edge = Discriminator(hidden_dim=36 * 36)
        
        self.encoder_adj = nn.Sequential(
            nn.Linear(768, 630),
            nn.Sigmoid()
        )
    
    def forward(self, feat, pos, sent):
        feat_seq, que_mask, x = self.lxrt_encoder(sent, (feat, pos))
        return feat_seq, que_mask, x
