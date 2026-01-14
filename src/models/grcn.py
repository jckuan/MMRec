# coding: utf-8
# 
"""
Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback, MM 2020
"""
import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from SAGEConv import SAGEConv
#from GATConv import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.utils.checkpoint import checkpoint

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from common.fettle_utils import initialize_fettle_losses, extract_cf_embeddings_average
##########################################################################

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='mean', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr='add')#, **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)


    def message(self,  x_i, x_j, size_i ,edge_index_i):
        #print(edge_index_i, x_i, x_j)
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        #print(self.alpha)
        #print(edge_index_i,size_i)
        # alpha = F.tanh(alpha)
        # self.alpha = F.leaky_relu(self.alpha)
        # alpha = torch.sigmoid(alpha)
        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
        # Sample attention coefficients stochastically.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j*self.alpha.view(-1,1)
        # return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out



class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm, use_checkpoint=True):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.use_checkpoint = use_checkpoint
        self.id_embedding = nn.Parameter( nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)         
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def _conv_forward(self, x, edge_index, weight_vector):
        """Separate method for checkpointing"""
        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)
        return x_hat_1, x_hat_2

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

        if self.has_norm:
            x = F.normalize(x)

        # Use gradient checkpointing for graph convolutions
        if self.use_checkpoint and self.training:
            x_hat_1, x_hat_2 = checkpoint(self._conv_forward, x, edge_index, weight_vector, use_reentrant=False)
        else:
            x_hat_1, x_hat_2 = self._conv_forward(x, edge_index, weight_vector)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(self, features, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, is_word=False, use_checkpoint=True, device=None):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.use_checkpoint = use_checkpoint
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word

        if is_word:
            self.word_tensor = torch.LongTensor(features).cuda()
            self.features = nn.Embedding(torch.max(features[1])+1, dim_C)
            nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = nn.Linear(self.dim_feat, self.dim_C)
            #print('MLP weight',self.MLP.weight)
            nn.init.xavier_normal_(self.MLP.weight)
            #print(self.MLP.weight)

    def _routing_forward(self, preference, features, edge_index_single):
        """Separate method for checkpointing routing iterations"""
        x = torch.cat((preference, features), dim=0)
        x_hat_1 = self.conv_embed_1(x, edge_index_single)
        return x_hat_1[:self.num_user]
    
    def _final_conv(self, x, edge_index_full):
        """Separate method for checkpointing final convolution"""
        x_hat_1 = self.conv_embed_1(x, edge_index_full)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        return x_hat_1

    def forward(self, edge_index):
        # MEMORY OPTIMIZATION: Transfer features from CPU to GPU if needed
        features = self.features
        if features.device != self.device:
            features = features.to(self.device)
        features = F.leaky_relu(self.MLP(features))
        
        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)
        else:
            preference = self.preference

        for i in range(self.num_routing):
            if self.use_checkpoint and self.training:
                update = checkpoint(self._routing_forward, preference, features, edge_index, use_reentrant=False)
            else:
                update = self._routing_forward(preference, features, edge_index)
            preference = preference + update

            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index_full = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

        if self.use_checkpoint and self.training:
            x_hat_1 = checkpoint(self._final_conv, x, edge_index_full, use_reentrant=False)
        else:
            x_hat_1 = self._final_conv(x, edge_index_full)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCN(GeneralRecommender):
    def __init__(self,  config, dataset):
        super(GRCN, self).__init__(config, dataset)
        
        # Store config for FETTLE
        self.config = config
        
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        dim_C = config['latent_embedding']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'add'
        self.weight_mode = 'confid'
        self.fusion_mode = 'concat'
        has_id = True
        has_act= False
        has_norm= True
        is_word = False
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']
        self.dropout = 0
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        #self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0
        self.pruning = True

        # Count modalities first
        num_model = 0
        if self.v_feat is not None:
            num_model += 1
        if self.t_feat is not None:
            num_model += 1

        use_checkpoint = config['use_checkpoint']  # Enable gradient checkpointing
        self.id_gcn = EGCN(num_user, num_item, dim_x, self.aggr_mode, has_act, has_norm, use_checkpoint)
        
        # Create modal-specific GCNs with checkpointing
        if self.v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm, use_checkpoint=use_checkpoint, device=self.device)
        if self.t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm, is_word, use_checkpoint, self.device)
        
        # Initialize model_specific_conf after counting modalities
        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, num_model))))

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).to(self.device)
        
        # FETTLE integration
        self.iladt_loss, self.cla_loss = initialize_fettle_losses(config, dim_x)
        
        
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))


    def forward(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
        #print('edge_index: ', edge_index)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep
            #print('weight_v is: ', weight)
            #print('content_rep: ',content_rep)

        #if self.a_feat is not None:
            #num_modal += 1
            #a_rep, weight_a = self.a_gcn(edge_index)
            #if weight is  None:
                #weight = weight_a  
                #content_rep = a_rep
            #else:
                #content_rep = torch.cat((content_rep,a_rep),dim=1)
                #if self.weight_mode == 'mean':
                    #weight = weight+ weight_a
                #else:
                    #weight = torch.cat((weight, weight_a), dim=1)

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t   
                conetent_rep = t_rep
            else:
                content_rep = torch.cat((content_rep,t_rep),dim=1)
                if self.weight_mode == 'mean':  
                    weight  = weight+  weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)   

        if self.weight_mode == 'mean':
        	weight = weight/num_modal

        elif self.weight_mode == 'max':
        	weight, _ = torch.max(weight, dim=1)
        	weight = weight.view(-1, 1)
            
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
            #print('weight is: ', weight)
            

        if self.pruning:
            weight = torch.relu(weight)
            


        id_rep = self.id_gcn(edge_index, weight)
        #print('id_rep is: ',id_rep)

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
            
        elif self.fusion_mode  == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep+v_rep+a_rep+t_rep)/4

        self.result = representation
        #print('representation is: ',representation)
        return representation
    
    def get_cf_embeddings(self):
        """Extract CF embeddings using average method for FETTLE"""
        # Get modality-specific embeddings after GCN
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
        
        v_rep = None
        t_rep = None
        
        if self.v_feat is not None:
            v_rep, _ = self.v_gcn(edge_index)
            v_rep = v_rep[self.n_users:]  # Extract item embeddings only
        
        if self.t_feat is not None:
            t_rep, _ = self.t_gcn(edge_index)
            t_rep = t_rep[self.n_users:]  # Extract item embeddings only
        
        # Average the embeddings if both exist
        if v_rep is not None and t_rep is not None:
            return extract_cf_embeddings_average(v_rep, t_rep)
        elif v_rep is not None:
            return v_rep
        elif t_rep is not None:
            return t_rep
        else:
            return None

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_gcn.id_embedding[user_tensor]**2 + self.id_gcn.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        reg_content_loss = torch.zeros(1).cuda() 
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor]**2).mean()
        #if self.a_feat is not None:
            #reg_content_loss = reg_content_loss + (self.a_gcn.preference[user_tensor]**2).mean()
        if self.t_feat is not None:            
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor]**2).mean()

        reg_confid_loss = (self.model_specific_conf**2).mean()
        
        reg_loss = reg_embedding_loss + reg_content_loss

        reg_loss = self.reg_weight * reg_loss
        #print('loss',loss + reg_loss)

        # FETTLE losses
        if self.config['use_fettle'] and self.iladt_loss is not None and self.cla_loss is not None:
            from common.fettle_utils import compute_fettle_losses, prepare_fettle_embeddings
            
            # Get CF embeddings
            cf_embeddings = self.get_cf_embeddings()
            
            if cf_embeddings is not None:
                # Get modality-specific embeddings
                edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
                v_rep, _ = self.v_gcn(edge_index) if self.v_feat is not None else (None, None)
                t_rep, _ = self.t_gcn(edge_index) if self.t_feat is not None else (None, None)
                
                # Extract item embeddings
                v_emb = v_rep[self.n_users:] if v_rep is not None else torch.zeros_like(cf_embeddings)
                t_emb = t_rep[self.n_users:] if t_rep is not None else torch.zeros_like(cf_embeddings)
                
                # Get user embeddings from result
                user_emb = self.result[:self.n_users]
                
                # Normalize embeddings
                import torch.nn.functional as F
                user_emb = F.normalize(user_emb, dim=1)
                cf_embeddings = F.normalize(cf_embeddings, dim=1)
                v_emb = F.normalize(v_emb, dim=1)
                t_emb = F.normalize(t_emb, dim=1)
                
                # Compute FETTLE losses
                iladt_loss_value, cla_loss_value = compute_fettle_losses(
                    self.iladt_loss, self.cla_loss,
                    user_emb, cf_embeddings, v_emb, t_emb,
                    batch_users, pos_items - self.n_users, self.config
                )
                
                # Add FETTLE losses to total loss
                loss = loss + self.config['iladt_weight'] * iladt_loss_value + self.config['cla_weight'] * cla_loss_value

        return loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


