# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']
        
        # Item popularity for popularity-based candidate sampling
        self.item_popularity_probs = None

        # Mixed Precision Training setup (matching DRAGON implementation)
        self.use_amp = config['use_amp'] if 'use_amp' in config else True
        # GradScaler with growth_interval to prevent NaN (wait longer before increasing scale)
        self.scaler = GradScaler('cuda', growth_interval=2000) if self.use_amp and self.device.type == 'cuda' else None
        if self.use_amp and self.device.type == 'cuda':
            self.logger.info('Mixed Precision Training (FP16) enabled - expect ~40-50% memory reduction')
            self.logger.info('GradScaler configured with growth_interval=2000 to prevent NaN')
        
        # PyTorch 2.0+ Compilation for ~20-30% speedup
        use_torch_compile = config['use_torch_compile'] if 'use_torch_compile' in config else False
        if hasattr(torch, 'compile') and use_torch_compile:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            self.logger.info('✅ Model compiled with torch.compile - expect ~20-30% speedup after first epoch')

        timestamp = datetime.now().strftime("%y%m%d-%H%M")
        self.writer = SummaryWriter(log_dir=f'runs/{config["dataset"]}_{timestamp}')

    def _compute_item_popularity(self, train_data):
        r"""Compute item popularity from training data for popularity-based sampling.
        
        Args:
            train_data (DataLoader): training data loader
        
        Returns:
            torch.Tensor: probability distribution over items based on training frequency
        """
        if self.tot_item_num is None:
            self.logger.warning('tot_item_num not set, cannot compute item popularity')
            return None
        
        item_counts = torch.zeros(self.tot_item_num, dtype=torch.float32, device=self.device)
        
        # Count item occurrences in training data
        for batched_data in train_data:
            # batched_data structure: [user_ids, item_ids, neg_ids (optional), ...]
            # We want to count positive items from training
            if isinstance(batched_data, torch.Tensor):
                if batched_data.dim() >= 2 and batched_data.size(0) >= 2:
                    item_indices = batched_data[1]  # Positive item IDs
                    unique_items, counts = torch.unique(item_indices, return_counts=True)
                    item_counts[unique_items] += counts.float()
            elif isinstance(batched_data, (list, tuple)) and len(batched_data) >= 2:
                item_indices = batched_data[1]
                if isinstance(item_indices, torch.Tensor):
                    unique_items, counts = torch.unique(item_indices, return_counts=True)
                    item_counts[unique_items] += counts.float()
        
        # Convert to probabilities with scaled Laplace smoothing
        # Use smaller smoothing parameter (alpha) when item catalog is large
        # to avoid over-smoothing and tiny probabilities
        alpha = 0.01 if self.tot_item_num > 10000 else 1.0
        item_popularity = (item_counts + alpha) / (item_counts.sum() + alpha * self.tot_item_num)
        
        self.logger.info(f'Computed item popularity: {(item_counts > 0).sum().item()}/{self.tot_item_num} items in training')
        self.logger.info(f'Smoothing parameter alpha={alpha}, min_prob={item_popularity.min().item():.2e}, max_prob={item_popularity.max().item():.2e}')
        return item_popularity
    
    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            
            # Mixed Precision Training (matching DRAGON implementation)
            if self.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    losses = loss_func(interaction)
                    if isinstance(losses, tuple):
                        loss = sum(losses)
                        loss_tuple = tuple(per_loss.item() for per_loss in losses)
                        total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                        for i, l in enumerate(loss_tuple):
                            self.writer.add_scalar(f'Loss/Train_loss_{i+1}', l, epoch_idx)
                    else:
                        loss = losses
                        total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                        self.writer.add_scalar('Loss/Train', loss.item(), epoch_idx)
                
                self._check_nan(loss)
                
                if self.mg and batch_idx % self.beta == 0:
                    # MG training with AMP
                    first_loss = self.alpha1 * loss
                    self.scaler.scale(first_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    with autocast('cuda'):
                        losses = loss_func(second_inter)
                        if isinstance(losses, tuple):
                            loss = sum(losses)
                        else:
                            loss = losses
                    
                    self._check_nan(loss)
                    second_loss = -1 * self.alpha2 * loss
                    self.scaler.scale(second_loss).backward()
                else:
                    # Standard training with AMP
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Standard FP32 training
                losses = loss_func(interaction)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                    for i, l in enumerate(loss_tuple):
                        self.writer.add_scalar(f'Loss/Train_loss_{i+1}', l, epoch_idx)
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                    self.writer.add_scalar('Loss/Train', loss.item(), epoch_idx)
                
                self._check_nan(loss)
                
                if self.mg and batch_idx % self.beta == 0:
                    # MG training without AMP
                    first_loss = self.alpha1 * loss
                    first_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    losses = loss_func(second_inter)
                    if isinstance(losses, tuple):
                        loss = sum(losses)
                    else:
                        loss = losses
                    
                    self._check_nan(loss)
                    second_loss = -1 * self.alpha2 * loss
                    second_loss.backward()
                else:
                    # Standard training without AMP
                    loss.backward()
                    if self.clip_grad_norm:
                        clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
                    self.optimizer.step()
            
            loss_batches.append(loss.detach())
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, epoch_idx=0):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data
            epoch_idx (int): current epoch index

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, epoch_idx=epoch_idx)
        
        # Determine which metric to use for early stopping
        # If candidate sampling is enabled, prioritize sampled metrics for validation score
        use_candidate_sampling = self.config['use_candidate_sampling']
        sampling_type = self.config['candidate_sampling_type']
        
        if use_candidate_sampling and sampling_type != 'none':
            # Try to find the validation metric with sampling suffix
            if sampling_type == 'uniform' or sampling_type == 'both':
                metric_key = f'{self.valid_metric}_uniform'
            elif sampling_type == 'popularity':
                metric_key = f'{self.valid_metric}_pop'
            else:
                metric_key = self.valid_metric
                
            # Fallback to base metric or default if not found
            if metric_key not in valid_result:
                if self.valid_metric in valid_result:
                    metric_key = self.valid_metric
                elif 'Recall@20_full' in valid_result:
                    metric_key = 'Recall@20_full'
                else:
                    metric_key = list(valid_result.keys())[0]  # Use first available metric
        else:
            metric_key = self.valid_metric if self.valid_metric in valid_result else 'NDCG@20'
            
        valid_score = valid_result[metric_key]
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = f"Epoch {epoch_idx} training [time: {e_time - s_time:.2f}s, "
        if isinstance(losses, tuple):
            train_loss_output = ', '.join(f'Train_loss{idx + 1}: {loss:.4f}' for idx, loss in enumerate(losses))
        else:
            train_loss_output += f'Train loss: {losses:.4f}'
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        # Initialize tot_item_num from dataset
        if self.tot_item_num is None and hasattr(train_data, 'dataset'):
            self.tot_item_num = train_data.dataset.item_num
            self.logger.info(f'Initialized tot_item_num: {self.tot_item_num}')
        
        # Compute item popularity once before training if popularity-based sampling is enabled
        use_candidate_sampling = self.config['use_candidate_sampling']
        sampling_type = self.config['candidate_sampling_type']
        if use_candidate_sampling and sampling_type in ['popularity', 'both']:
            if self.item_popularity_probs is None:
                self.logger.info('Computing item popularity from training data...')
                self.item_popularity_probs = self._compute_item_popularity(train_data)
        
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_score, valid_result = self._valid_epoch(valid_data, epoch_idx)
                # Log metrics to tensorboard
                for metric, value in valid_result.items():
                    self.writer.add_scalar(f'Metrics/Valid_{metric}', value, epoch_idx)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                # test
                _, test_result = self._valid_epoch(test_data, epoch_idx)
                # Log test metrics to tensorboard
                for metric, value in test_result.items():
                    self.writer.add_scalar(f'Metrics/Test_{metric}', value, epoch_idx)
                
                if verbose:
                    self.logger.info(f'Epoch {epoch_idx}: valid_score: {valid_score:.4f}')
                    self.logger.info('Valid result: ' + dict2str(valid_result))
                    self.logger.info('Test result : ' + dict2str(test_result))
                    
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        # Close tensorboard writer
        self.writer.close()
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0, epoch_idx=0):
        r"""Evaluate the model based on the eval data.
        
        Args:
            eval_data: evaluation data loader
            is_test: whether this is test evaluation
            idx: evaluation index
            epoch_idx: current epoch index (used to skip full-rank after epoch 0)
            
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()
        
        # Get candidate sampling configuration
        use_candidate_sampling = self.config['use_candidate_sampling']
        sampling_type = self.config['candidate_sampling_type']
        num_neg_candidates = self.config['num_negative_candidates']
        
        # If candidate sampling is disabled or type is 'none', do full-rank evaluation
        if not use_candidate_sampling or sampling_type == 'none':
            return self._evaluate_full_rank(eval_data, is_test, idx)
        
        # Determine which sampling methods to use
        use_uniform = sampling_type in ['uniform', 'both']
        use_popularity = sampling_type in ['popularity', 'both']
        
        results = {}
        
        # Only compute full-rank at epoch 0 for baseline reference
        # Skip in later epochs to save computation time
        if epoch_idx == 0:
            full_rank_results = self._evaluate_full_rank(eval_data, is_test, idx)
            for key, value in full_rank_results.items():
                results[f'{key}_full'] = value
        
        # Uniform sampling
        if use_uniform:
            uniform_results = self._evaluate_with_sampling(eval_data, is_test, idx, 
                                                           num_neg_candidates, 'uniform')
            for key, value in uniform_results.items():
                results[f'{key}_uniform'] = value
        
        # Popularity-based sampling
        if use_popularity:
            popularity_results = self._evaluate_with_sampling(eval_data, is_test, idx,
                                                              num_neg_candidates, 'popularity')
            for key, value in popularity_results.items():
                results[f'{key}_pop'] = value
        
        return results
    
    def _evaluate_full_rank(self, eval_data, is_test=False, idx=0):
        r"""Full-rank evaluation (original method)."""

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # Use mixed precision for evaluation (matching DRAGON implementation)
            if self.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    scores = self.model.full_sort_predict(batched_data)
                # Convert to FP32 for masking to avoid overflow with large negative values
                scores = scores.float()
            else:
                scores = self.model.full_sort_predict(batched_data)
            
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)
    
    def _evaluate_with_sampling(self, eval_data, is_test=False, idx=0, num_neg_candidates=99, sampling_type='uniform'):
        r"""Candidate sampling evaluation with uniform or popularity-based negative sampling.
        
        For each test user:
        1. Keep all TEST POSITIVE items in the candidate set
        2. Sample num_neg_candidates negative items (not in training, not in test)
        3. Score only candidates: test_positives + sampled_negatives
        4. Compute metrics on this reduced candidate set
        
        Args:
            eval_data: evaluation data loader
            is_test: whether this is test evaluation
            idx: evaluation index
            num_neg_candidates: number of negative items to sample per user
            sampling_type: 'uniform' or 'popularity'
        """
        # Get item popularity for popularity-based sampling
        item_popularity_probs = None
        if sampling_type == 'popularity':
            if self.item_popularity_probs is not None:
                item_popularity_probs = self.item_popularity_probs
            else:
                # Fallback to uniform if popularity not computed
                n_items = self.tot_item_num if self.tot_item_num is not None else eval_data.dataset.item_num
                item_popularity_probs = torch.ones(n_items, device=self.device) / n_items
                self.logger.warning('Item popularity not computed, using uniform distribution as fallback')
        
        # Get test positive items for each user
        pos_items_list = eval_data.get_eval_items()
        
        batch_matrix_list = []
        user_offset = 0
        
        for batch_idx, batched_data in enumerate(eval_data):
            # Get full scores
            if self.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    scores = self.model.full_sort_predict(batched_data)
                scores = scores.float()
            else:
                scores = self.model.full_sort_predict(batched_data)
            
            masked_items = batched_data[1]  # Training items
            batch_size = scores.size(0)
            n_items = scores.size(1)
            
            # Create candidate-sampled scores (mask out non-candidates)
            sampled_scores = torch.full_like(scores, -1e10)
            
            for user_idx in range(batch_size):
                global_user_idx = user_offset + user_idx
                
                # Get user's training items (to exclude from candidate sampling)
                user_mask = masked_items[0] == user_idx
                user_train_items = masked_items[1][user_mask]
                
                # Get user's TEST POSITIVE items (MUST be included in candidates)
                user_test_items = pos_items_list[global_user_idx]
                if isinstance(user_test_items, (list, np.ndarray)):
                    user_test_items = torch.tensor(user_test_items, device=self.device)
                elif not isinstance(user_test_items, torch.Tensor):
                    user_test_items = torch.tensor([user_test_items], device=self.device)
                
                # CRITICAL: Always include test positives in candidate set
                sampled_scores[user_idx, user_test_items] = scores[user_idx, user_test_items]
                
                # Available items for negative sampling = all items - training - test
                available_mask = torch.ones(n_items, dtype=torch.bool, device=self.device)
                available_mask[user_train_items] = False
                available_mask[user_test_items] = False  # Exclude test items from negative sampling
                available_items = torch.where(available_mask)[0]
                
                if len(available_items) == 0:
                    continue
                
                # Sample negative candidates
                num_to_sample = min(num_neg_candidates, len(available_items))
                
                if sampling_type == 'popularity' and item_popularity_probs is not None:
                    # Popularity-based sampling
                    probs = item_popularity_probs[available_items]
                    probs = probs / probs.sum()
                    sampled_indices = torch.multinomial(probs, num_to_sample, replacement=False)
                    sampled_items = available_items[sampled_indices]
                else:
                    # Uniform sampling
                    perm = torch.randperm(len(available_items), device=self.device)[:num_to_sample]
                    sampled_items = available_items[perm]
                
                # Add sampled negatives to candidate set
                sampled_scores[user_idx, sampled_items] = scores[user_idx, sampled_items]
            
            # Get top-k from candidates only
            _, topk_index = torch.topk(sampled_scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
            
            user_offset += batch_size
        
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

