"""
FETTLE Utility Functions
========================
Helper functions for FETTLE (Feedback-Oriented Multi-Modal Alignment) integration.
These utilities minimize code duplication across different model implementations.

Paper: "Who To Align With: Feedback-Oriented Multi-Modal Alignment in Recommendation Systems" (SIGIR 2024)
"""

import torch
import torch.nn as nn
from .loss import CLALoss, ILADTLoss


def initialize_fettle_losses(config, embedding_dim):
    """
    Initialize FETTLE loss modules based on configuration.
    
    Args:
        config: Configuration object with FETTLE parameters
        embedding_dim (int): Dimension of embeddings
    
    Returns:
        tuple: (iladt_loss, cla_loss) - Initialized loss modules or (None, None)
    """
    if not config['use_fettle']:
        return None, None
    
    # Initialize ILA+DT Loss
    iladt_loss = ILADTLoss(
        dim=embedding_dim,
        gamma=config['clcr_gamma']  # Temperature parameter
    )
    
    # Initialize CLA Loss
    cla_loss = CLALoss(
        K=config['prototype_num'],  # Number of prototypes
        D=embedding_dim,
        gamma=config['ga_gamma']  # Temperature parameter
    )
    
    return iladt_loss, cla_loss


def extract_cf_embeddings_average(v_rep, t_rep):
    """
    Extract collaborative filtering embeddings using average method.
    
    This is the simplest CF extraction method that averages visual and text embeddings
    after GCN propagation. More sophisticated methods (learnable fusion, separate
    embeddings) are available in Dragon-for-Music but not needed for basic integration.
    
    Args:
        v_rep: Visual embeddings [num_items, dim]
        t_rep: Text embeddings [num_items, dim]
    
    Returns:
        Averaged CF embeddings [num_items, dim]
    """
    return (v_rep + t_rep) / 2


def prepare_fettle_embeddings(all_users, all_items, v_rep, t_rep, 
                               num_users, num_items, batch_user, batch_pos_item):
    """
    Prepare embeddings for FETTLE loss computation.
    
    Handles embedding extraction, squeezing, and index conversion from global
    item indices to item-only indices (required by FETTLE losses).
    
    Args:
        all_users: User embeddings from GCN [num_users, 1, dim] or [num_users, dim]
        all_items: Item CF embeddings [num_items, 1, dim] or [num_items, dim]
        v_rep: Visual embeddings [num_items, 1, dim] or [num_items, dim]
        t_rep: Text embeddings [num_items, 1, dim] or [num_items, dim]
        num_users: Total number of users
        num_items: Total number of items
        batch_user: Batch user indices [batch_size]
        batch_pos_item: Batch positive item indices (global) [batch_size]
    
    Returns:
        tuple: (user_emb, item_emb, v_emb, t_emb, batch_user, batch_pos_item_local)
            - All embeddings have shape [num, dim] (squeezed)
            - batch_pos_item_local: Item indices converted to item-only space
    """
    # Squeeze embeddings if they have extra dimension
    user_emb = all_users[:num_users].squeeze() if all_users.dim() == 3 else all_users[:num_users]
    item_emb = all_items.squeeze() if all_items.dim() == 3 else all_items
    v_emb = v_rep.squeeze() if v_rep.dim() == 3 else v_rep
    t_emb = t_rep.squeeze() if t_rep.dim() == 3 else t_rep
    
    # Convert global item indices to item-only indices
    # batch_pos_item is in range [0, num_users+num_items)
    # Need to convert to [0, num_items) by subtracting num_users
    batch_pos_item_local = batch_pos_item - num_users
    
    return user_emb, item_emb, v_emb, t_emb, batch_user, batch_pos_item_local


def compute_fettle_losses(iladt_loss, cla_loss, user_emb, item_emb, v_emb, t_emb,
                          batch_user, batch_pos_item_local, config, epoch_idx=None):
    """
    Compute FETTLE losses with NaN checking.
    
    Args:
        iladt_loss: ILA+DT loss module
        cla_loss: CLA loss module
        user_emb: User embeddings [num_users, dim]
        item_emb: Item CF embeddings [num_items, dim]
        v_emb: Visual embeddings [num_items, dim]
        t_emb: Text embeddings [num_items, dim]
        batch_user: Batch user indices [batch_size]
        batch_pos_item_local: Batch item indices (item-only space) [batch_size]
        config: Configuration dict with loss weights
        epoch_idx: Optional epoch index
    
    Returns:
        tuple: (iladt_loss_value, cla_loss_value)
            - Returns 0 for losses that result in NaN
    """
    # Compute ILA+DT Loss
    iladt_loss_value = iladt_loss(
        user_emb, item_emb, v_emb, t_emb,
        batch_user, batch_pos_item_local,
        epoch_idx=epoch_idx
    )
    
    # Compute CLA Loss
    cla_loss_value = cla_loss(
        user_emb, item_emb, v_emb, t_emb,
        batch_user, batch_pos_item_local,
        mode='SwAV'
    )
    
    # NaN checking
    if torch.isnan(iladt_loss_value):
        iladt_loss_value = torch.tensor(0.0, device=iladt_loss_value.device)
    if torch.isnan(cla_loss_value):
        cla_loss_value = torch.tensor(0.0, device=cla_loss_value.device)
    
    return iladt_loss_value, cla_loss_value
