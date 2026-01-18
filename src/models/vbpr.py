# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
from common.fettle_utils import initialize_fettle_losses
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # Store config for FETTLE
        self.config = config

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Memory optimization: gradient checkpointing
        self.use_checkpoint = config['use_checkpoint']

        # FETTLE integration - need separate linear layers for each modality
        self.iladt_loss, self.cla_loss = initialize_fettle_losses(config, self.i_embedding_size)
        if self.iladt_loss is not None:
            # Create separate linear layers for FETTLE
            if self.v_feat is not None:
                self.v_linear = nn.Linear(self.v_feat.shape[1], self.i_embedding_size)
            if self.t_feat is not None:
                self.t_linear = nn.Linear(self.t_feat.shape[1], self.i_embedding_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def _item_linear_transform(self, features):
        """Helper for checkpointing linear transformation"""
        return self.item_linear(features)
    
    def forward(self, dropout=0.0):
        # Memory optimization: use gradient checkpointing for expensive linear transformation
        if self.use_checkpoint and self.training:
            # Transfer features from CPU to GPU if needed
            features = self.item_raw_features.to(self.device) if self.item_raw_features.device != self.device else self.item_raw_features
            item_embeddings = checkpoint(self._item_linear_transform, features, use_reentrant=False)
        else:
            features = self.item_raw_features.to(self.device) if self.item_raw_features.device != self.device else self.item_raw_features
            item_embeddings = self.item_linear(features)
        
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        # Memory optimization: in-place dropout
        user_e = F.dropout(self.u_embedding, dropout, inplace=True) if dropout > 0 else self.u_embedding
        item_e = F.dropout(item_embeddings, dropout, inplace=True) if dropout > 0 else item_embeddings
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        
        # FETTLE losses
        if self.config['use_fettle'] and self.iladt_loss is not None and self.cla_loss is not None:
            from common.fettle_utils import compute_fettle_losses
            
            # Extract visual and text features for FETTLE using separate linear layers
            v_emb = self.v_linear(self.v_feat) if self.v_feat is not None and hasattr(self, 'v_linear') else torch.zeros_like(self.i_embedding)
            t_emb = self.t_linear(self.t_feat) if self.t_feat is not None and hasattr(self, 't_linear') else torch.zeros_like(self.i_embedding)
            
            # Use ID embeddings as CF representations (VBPR doesn't have separate CF embeddings)
            item_cf_emb = self.i_embedding
            
            # Prepare user embeddings (split VBPR's concatenated user embeddings)
            user_cf_emb = user_embeddings[:, :self.u_embedding_size]
            
            # Normalize embeddings
            user_cf_emb = F.normalize(user_cf_emb, dim=1)
            item_cf_emb = F.normalize(item_cf_emb, dim=1)
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            
            # Compute FETTLE losses
            iladt_loss_value, cla_loss_value = compute_fettle_losses(
                self.iladt_loss, self.cla_loss,
                user_cf_emb, item_cf_emb, v_emb, t_emb,
                user, pos_item, self.config
            )
            
            # Add FETTLE losses to total loss
            loss = loss + self.config['iladt_weight'] * iladt_loss_value + self.config['cla_weight'] * cla_loss_value
        
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
