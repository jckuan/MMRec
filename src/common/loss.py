# coding: utf-8
# @email  : enoche.chow@gmail.com


import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss


# ============================================================================
# FETTLE: Feedback-Oriented Multi-Modal Alignment Losses
# Paper: "Who To Align With: Feedback-Oriented Multi-Modal Alignment 
#         in Recommendation Systems" (SIGIR 2024)
# ============================================================================

import torch.nn.functional as F
from torch_scatter import scatter_add


class CLALoss(nn.Module):
    """
    Cluster-Level Alignment Loss (CLA)
    
    Aligns different modalities at the cluster/prototype level using SwAV-style
    clustering and cross-entropy loss.
    
    Args:
        K (int): Number of prototypes for each modality
        D (int): Dimension of each prototype
        gamma (float): Temperature parameter
    """
    def __init__(self, K: int, D: int, gamma: float) -> None:
        super(CLALoss, self).__init__()
        self.K = K
        self.D = D
        self.feat2code = nn.Linear(D, K, bias=False)
        self.ii2code = nn.Linear(D, K, bias=False)
        self.gamma = nn.Parameter(torch.ones([]) * gamma)

    def sinkhorn(self, out):
        """Sinkhorn-Knopp algorithm for optimal transport"""
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        
        sinkhorn_iterations = 3
        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, user_embeddings, id_embeddings, image_embeddings, text_embeddings, 
                users, items, mode='SwAV', user_code=None, item_code=None, 
                image_code=None, text_code=None):
        """
        Compute cluster-level alignment loss
        
        Args:
            user_embeddings: [num_users, D] - User CF embeddings
            id_embeddings: [num_items, D] - Item CF embeddings
            image_embeddings: [num_items, D] - Visual embeddings
            text_embeddings: [num_items, D] - Text embeddings
            users: [batch_size] - User indices
            items: [batch_size] - Item indices
            mode: 'SwAV' for Sinkhorn clustering, else use provided codes
        
        Returns:
            Scalar loss combining user-item and multi-modal alignment
        """
        with torch.no_grad():
            self.gamma.clamp_(0.01, 0.99)
        
        # Normalize embeddings
        user_embeddings = F.normalize(user_embeddings[users], dim=1)
        id_embeddings = F.normalize(id_embeddings[items], dim=1)
        image_embeddings = F.normalize(image_embeddings[items], dim=1)
        text_embeddings = F.normalize(text_embeddings[items], dim=1)

        # Normalize prototype weights
        with torch.no_grad():
            w = self.feat2code.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.feat2code.weight.copy_(w)
            w = self.ii2code.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.ii2code.weight.copy_(w)

        # Compute codes (assignments to prototypes)
        code_user = user_embeddings @ self.feat2code.weight.t()
        code_id = id_embeddings @ self.feat2code.weight.t()
        code_id_ii = id_embeddings @ self.ii2code.weight.t()
        code_image_ii = image_embeddings @ self.ii2code.weight.t()
        code_text_ii = text_embeddings @ self.ii2code.weight.t()

        # Get cluster assignments
        with torch.no_grad():
            if mode == 'SwAV':
                q_id = self.sinkhorn(code_id.detach())
                q_user = self.sinkhorn(code_user.detach())
                q_id_ii = self.sinkhorn(code_id_ii.detach())
                q_image_ii = self.sinkhorn(code_image_ii.detach())
                q_text_ii = self.sinkhorn(code_text_ii.detach())
            else:
                q_id = item_code.detach()
                q_user = user_code.detach()
                q_id_ii = q_id
                q_image_ii = image_code.detach()
                q_text_ii = text_code.detach()

        gamma = self.gamma
        if mode == 'SwAV':
            # User-Item alignment loss
            loss = 0
            loss += -torch.mean(torch.sum(q_user * F.log_softmax(code_id / gamma, dim=1), dim=1))
            loss += -torch.mean(torch.sum(q_id * F.log_softmax(code_user / gamma, dim=1), dim=1))

            # Multi-modal alignment loss
            align_loss = 0
            align_loss += -torch.mean(torch.sum(q_id_ii * F.log_softmax(code_image_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_id_ii * F.log_softmax(code_text_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_image_ii * F.log_softmax(code_id_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_image_ii * F.log_softmax(code_text_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_text_ii * F.log_softmax(code_id_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_text_ii * F.log_softmax(code_image_ii / gamma, dim=1), dim=1))
        else:
            loss = 0
            loss += F.cross_entropy(code_id[items], q_user[users])
            loss += F.cross_entropy(code_user[users], q_id[items])
            align_loss = 0
            align_loss += F.cross_entropy(code_image_ii[items], q_id_ii[items])
            align_loss += F.cross_entropy(code_text_ii[items], q_id_ii[items])
            align_loss += F.cross_entropy(code_id_ii[items], q_image_ii[items])
            align_loss += F.cross_entropy(code_text_ii[items], q_image_ii[items])
            align_loss += F.cross_entropy(code_text_ii[items], q_image_ii[items])
            align_loss += F.cross_entropy(code_id_ii[items], q_text_ii[items])
            align_loss += F.cross_entropy(code_image_ii[items], q_text_ii[items])

        return loss / 2 + align_loss / 6


class ILADTLoss(nn.Module):
    """
    Item-Level Alignment + Direction Tuning Loss (ILA+DT)
    
    Aligns modalities at the item level by:
    1. Identifying which modality better captures user preferences
    2. Learning directional transformations to align weaker modalities
    3. Ensuring transformations improve user preference scores
    
    Args:
        dim (int): Embedding dimension
        gamma (float): Initial temperature parameter (learnable)
    """
    def __init__(self, dim=64, gamma=0.007):
        super(ILADTLoss, self).__init__()
        self.temp = nn.Parameter(gamma * torch.ones([]))
        
        # Directional transformation layers
        self.i2t_map = nn.Linear(dim, dim, bias=False)  # Image to Text
        self.t2i_map = nn.Linear(dim, dim, bias=False)  # Text to Image
        self.i2d_map = nn.Linear(dim, dim, bias=False)  # Image to CF
        self.d2i_map = nn.Linear(dim, dim, bias=False)  # CF to Image
        self.t2d_map = nn.Linear(dim, dim, bias=False)  # Text to CF
        self.d2t_map = nn.Linear(dim, dim, bias=False)  # CF to Text

    def forward(self, user_embeddings, item_embeddings, image_embeddings, text_embeddings,
                user_id, item_id, epoch_idx=None):
        """
        Compute item-level alignment + direction tuning loss
        
        Args:
            user_embeddings: [num_users, dim] - User CF embeddings
            item_embeddings: [num_items, dim] - Item CF embeddings
            image_embeddings: [num_items, dim] - Visual embeddings
            text_embeddings: [num_items, dim] - Text embeddings
            user_id: [batch_size] - User indices
            item_id: [batch_size] - Item indices
            epoch_idx: Optional epoch index (unused)
        
        Returns:
            Scalar loss combining alignment and direction tuning
        """
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)

        # Sort items to group same items together
        item_id, indice = torch.sort(item_id)
        unique_items, remap_indexs, counts = torch.unique(
            item_id, return_inverse=True, return_counts=True, sorted=True
        )
        user_id = user_id[indice]

        # Normalize embeddings
        user_embeddings = user_embeddings / user_embeddings.norm(dim=1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)

        # Compute similarity scores for each modality
        uid_scores = torch.sum(user_embeddings[user_id] * item_embeddings[item_id], dim=1).squeeze()
        uii_scores = torch.sum(user_embeddings[user_id] * image_embeddings[item_id], dim=1).squeeze()
        uit_scores = torch.sum(user_embeddings[user_id] * text_embeddings[item_id], dim=1).squeeze()

        # Aggregate scores per unique item
        with torch.no_grad():
            num_unique = unique_items.shape[0]
            uid_scores = scatter_add(uid_scores / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            uii_scores = scatter_add(uii_scores / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            uit_scores = scatter_add(uit_scores / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)

            # Determine which modality is stronger (feedback-oriented)
            t_i_mask = uii_scores > uit_scores  # Image better than text
            i_t_mask = uit_scores > uii_scores  # Text better than image
            i_d_mask = uid_scores > uii_scores  # CF better than image
            d_i_mask = uii_scores > uid_scores  # Image better than CF
            t_d_mask = uid_scores > uit_scores  # CF better than text
            d_t_mask = uit_scores > uid_scores  # Text better than CF

        # Use unique items only
        item_embeddings = item_embeddings[unique_items]
        image_embeddings = image_embeddings[unique_items]
        text_embeddings = text_embeddings[unique_items]

        # Original normalized features
        image_features_norm = image_embeddings
        text_features_norm = text_embeddings
        cf_features_norm = item_embeddings

        # Apply directional transformations with residual connections
        image_features_i2t = self.i2t_map(image_embeddings) + image_embeddings
        image_features_i2d = self.i2d_map(image_embeddings) + image_embeddings
        text_features_t2i = self.t2i_map(text_embeddings) + text_embeddings
        text_features_t2d = self.t2d_map(text_embeddings) + text_embeddings
        cf_features_d2i = self.d2i_map(item_embeddings) + item_embeddings
        cf_features_d2t = self.d2t_map(item_embeddings) + item_embeddings

        # Normalize transformed features
        image_features_norm_i2t = image_features_i2t / image_features_i2t.norm(dim=1, keepdim=True)
        image_features_norm_i2d = image_features_i2d / image_features_i2d.norm(dim=1, keepdim=True)
        text_features_norm_t2i = text_features_t2i / text_features_t2i.norm(dim=1, keepdim=True)
        text_features_norm_t2d = text_features_t2d / text_features_t2d.norm(dim=1, keepdim=True)
        cf_features_norm_d2i = cf_features_d2i / cf_features_d2i.norm(dim=1, keepdim=True)
        cf_features_norm_d2t = cf_features_d2t / cf_features_d2t.norm(dim=1, keepdim=True)

        # Compute alignment logits
        logits_image_cf = (image_features_norm_i2d @ cf_features_norm.t().detach())[i_d_mask] / self.temp
        logits_cf_image = (cf_features_norm_d2i @ image_features_norm.t().detach())[d_i_mask] / self.temp
        logits_cf_text = (cf_features_norm_d2t @ text_features_norm.t().detach())[d_t_mask] / self.temp
        logits_text_cf = (text_features_norm_t2d @ cf_features_norm.t().detach())[t_d_mask] / self.temp
        logits_image_text = (image_features_norm_i2t @ text_features_norm.detach().t())[i_t_mask] / self.temp
        logits_text_image = (text_features_norm_t2i @ image_features_norm.detach().t())[t_i_mask] / self.temp

        # Item-level alignment loss
        loss = 0
        labels = torch.arange(unique_items.shape[0]).to(logits_image_cf.device)

        if t_i_mask.sum() > 0:
            t2i_loss = F.cross_entropy(logits_text_image, labels[t_i_mask], reduction='sum')
            loss += t2i_loss
        if i_t_mask.sum() > 0:
            i2t_loss = F.cross_entropy(logits_image_text, labels[i_t_mask], reduction='sum')
            loss += i2t_loss
        if i_d_mask.sum() > 0:
            i2d_loss = F.cross_entropy(logits_image_cf, labels[i_d_mask], reduction='sum')
            loss += i2d_loss
        if d_i_mask.sum() > 0:
            d2i_loss = F.cross_entropy(logits_cf_image, labels[d_i_mask], reduction='sum')
            loss += d2i_loss
        if t_d_mask.sum() > 0:
            t2d_loss = F.cross_entropy(logits_text_cf, labels[t_d_mask], reduction='sum')
            loss += t2d_loss
        if d_t_mask.sum() > 0:
            d2t_loss = F.cross_entropy(logits_cf_text, labels[d_t_mask], reduction='sum')
            loss += d2t_loss
        
        loss = loss / unique_items.shape[0]

        # Direction tuning loss
        loss_align = 0
        user_embeddings = user_embeddings.detach()
        num_unique = unique_items.shape[0]
        
        if i_t_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * image_features_norm_i2t[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[i_t_mask] - uii_scores[i_t_mask])
        if t_i_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * text_features_norm_t2i[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[t_i_mask] - uit_scores[t_i_mask])
        if i_d_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * image_features_norm_i2d[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[i_d_mask] - uii_scores[i_d_mask])
        if d_i_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * cf_features_norm_d2i[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[d_i_mask] - uid_scores[d_i_mask])
        if t_d_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * text_features_norm_t2d[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[t_d_mask] - uit_scores[t_d_mask])
        if d_t_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * cf_features_norm_d2t[remap_indexs], dim=1).squeeze()
            pos_score = scatter_add(pos_score / counts[remap_indexs], remap_indexs, dim=0, dim_size=num_unique)
            loss_align += -torch.mean(pos_score[d_t_mask] - uid_scores[d_t_mask])
        
        loss_align = loss_align / 6
        
        total_loss = loss + loss_align
        total_loss = torch.clamp(total_loss, min=-100, max=100)
        
        return total_loss