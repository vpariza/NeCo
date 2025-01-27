# Some methods adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import distributed as dist
from torch import nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torchvision.ops import roi_align
from typing import Callable, Optional, List, Any, Iterator, Tuple, Dict

from experiments.utils import PredsmIoUKmeans, process_attentions, cosine_scheduler
from src.models.vit import vit_small, vit_base, vit_large
from src.models.vit_v2 import vit_small as vit_small_v2, vit_base as vit_base_v2, vit_large as vit_large_v2

from diffsort import DiffSortNet
import re

class NeCo(pl.LightningModule):

    def __init__(self, gpus: int, num_samples: int, batch_size: int, max_epochs: int, lr_heads: float,
                 lr_backbone: float, final_lr: float, weight_decay_end: float, weight_decay: float, 
                 projection_hidden_dim: int, projection_feat_dim: int, n_layers_projection_head: int, 
                 crops_for_assign: List[int], nmb_crops: List[int], num_classes: int, val_iters: int,
                 num_clusters_kmeans: List[int], use_teacher: bool = True, loss_mask: str = 'all',
                 queue_length: int = 0, momentum_teacher: float = 0.9995, momentum_teacher_end: float = 1.,
                 exclude_norm_bias: bool = True, optimizer: str = 'adam', num_nodes: int = 1,
                 patch_size: int = 16, roi_align_kernel_size: int = 7, val_downsample_masks: bool = True,
                 arch: str = 'vit-small', student_steepness=5, teacher_steepness=None, 
                 arch_version = 'v1', grad_norm_clipping=None,
                 sort_net='bitonic', trainable_blocks= None,
                 is_queue_usable=True, sim_dist='euclidean'):
        """
        Initializes the NeCo for training. We use pytorch lightning as framework.
        :param gpus: number of gpus used per node
        :param num_samples: number of samples in train data
        :param batch_size: batch size per GPU
        :param max_epochs: the number of epochs
        :param lr_heads: learning rate for clustering projection head
        :param lr_backbone: learning rate for ViT backbone
        :param final_lr: final learning rate for cosine learning rate schedule
        :param weight_decay_end: final weight decay for cosine weight decay schedule
        :param weight_decay: weight decay for optimizer
        :param projection_hidden_dim: embedding dimensionality of hidden layers in projection head
        :param projection_feat_dim: embedding dimensionality of output layer in projection head
        :param n_layers_projection_head: number of layers for projection head
        :param crops_for_assign: list of crop ids for computing optimal cluster assignment
        :param nmb_crops: number of global and local crops to be used during training
        :param num_classes: number of gt classes of validation data
        :param val_iters: number of validation iterations per epoch.
        :param num_clusters_kmeans: list of clustering granularities to be used to evaluate learnt feature space
        :param use_teacher: flag to indicate whether a teacher network should be used for computing the optimal cluster
        assignments
        :param loss_mask: indicates masking mode for computing cross entropy. Choose from 'fg', 'all' and 'bg'.
        :param queue_length: length of queue.
        :param momentum_teacher: start value of momentum for teacher network
        :param momentum_teacher_end: end value of momentum for teacher network
        :param exclude_norm_bias: flag to exclude norm and bias from weight decay
        :param optimizer: type of optimizer to use. Currently only supports adamw
        :param num_nodes: number of nodes to train on
        :param patch_size: patch size used for vision transformer
        :param roi_align_kernel_size: kernel size to be used for aligning the predicted and optimal spatial outputs
        each crop's bounding box.
        :param val_downsample_masks: flag to downsample masks for evaluation. If set mIoU is evaluated on 100x100 masks.
        :param arch: architecture of model to be fine-tuned. Currently supports vit-small, vit-base and vit-large.
        :param sort_net: sorting network to be used for sorting. Currently supports 'bitonic' and 'odd_even'.
        :param trainable_blocks: list of blocks to be trained. If None, all blocks are trained.
        :param is_queue_usable: flag to indicate whether queue should be used for selecting reference patches.
        :param sim_dist: similarity distance to be used for computing similarity between patches. Choose from 'euclidean' and 'cosine'.
        :param student_steepness: steepness of student network for sorting
        :param teacher_steepness: steepness of teacher network for sorting. If None, student_steepness is used.
        :param arch_version: version of architecture to be used. Currently supports 'v1' and 'v2'.
        :param grad_norm_clipping: gradient norm clipping value. If None, no clipping is applied.
        """
        super().__init__()
        self.save_hyperparameters()
        self.roi_align_kernel_size = roi_align_kernel_size
        self.lr_heads = lr_heads
        self.patch_size = patch_size
        self.projection_hidden_dim = projection_hidden_dim
        self.n_layers_projection_head = n_layers_projection_head
        self.val_downsample_masks = val_downsample_masks
        self.arch = arch
        self.gpus = gpus
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.projection_feat_dim = projection_feat_dim
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.optim = optimizer
        self.exclude_norm_bias = exclude_norm_bias
        self.weight_decay = weight_decay
        self.final_lr = final_lr
        self.lr_backbone = lr_backbone
        self.max_epochs = max_epochs
        self.val_iters = val_iters
        self.num_clusters_kmeans = num_clusters_kmeans
        self.num_classes = num_classes
        self.loss_mask = loss_mask
        self.use_teacher = use_teacher
        self.arch_version = arch_version
        self.grad_norm_clipping = grad_norm_clipping
        self.sort_net = sort_net
        self.entries_in_queue = 0

        # diff sorting params
        self.student_steepness = student_steepness
        self.teacher_steepness = teacher_steepness

        self.trainable_blocks = trainable_blocks
        self.is_queue_usable = is_queue_usable
        self.sim_dist = sim_dist

        # queue params
        self.queue_length = queue_length
        self.queue = None

        # init model
        if self.use_teacher:
            self.teacher = None
        self.model = self.init_model()  # inits teacher as well
        self.softmax = nn.Softmax(dim=1)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # init wd and momentum schedule
        self.wd_schedule = cosine_scheduler(self.weight_decay, weight_decay_end,
                                            self.max_epochs, self.train_iters_per_epoch)
        if self.use_teacher:
            self.momentum_schedule = cosine_scheduler(momentum_teacher, momentum_teacher_end,
                                                      self.max_epochs, self.train_iters_per_epoch)

        # init metric modules
        self.preds_miou_layer4 = PredsmIoUKmeans(num_clusters_kmeans, num_classes)

    def init_model(self):
        # Initialize model and optionally the teacher
        if self.arch == 'vit-small':
            if self.arch_version == 'v2':
                model_func = vit_small_v2
            else:
                model_func = vit_small
        elif self.arch == 'vit-base':
            if self.arch_version == 'v2':
                model_func = vit_base_v2
            else:
                model_func = vit_base
        elif self.arch == 'vit-large':
            if self.arch_version == 'v2':
                model_func = vit_large_v2
            else:                
                model_func = vit_large
        else:
            raise ValueError(f"{self.arch} is not supported")
        if self.use_teacher:
            self.teacher = model_func(patch_size=self.patch_size,
                                      output_dim=self.projection_feat_dim,
                                      hidden_dim=self.projection_hidden_dim,
                                      n_layers_projection_head=self.n_layers_projection_head)
        return model_func(patch_size=self.patch_size,
                         output_dim=self.projection_feat_dim,
                         hidden_dim=self.projection_hidden_dim,
                         n_layers_projection_head=self.n_layers_projection_head)

    def on_train_epoch_start(self):
        # Init queue if queue is None
        if self.queue_length > 0 and self.queue is None:
            self.queue = torch.zeros(
                len(self.crops_for_assign),
                self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                self.projection_feat_dim,
                )
            if self.gpus > 0:
                self.queue = self.queue.cuda()

        self.use_the_queue = False
        
        self.student_sorter = DiffSortNet(self.sort_net, self.roi_align_kernel_size**2, device=self.device, steepness=self.student_steepness)
        if self.teacher_steepness is not None:
            self.teacher_sorter = DiffSortNet(self.sort_net, self.roi_align_kernel_size**2, device=self.device, steepness=self.teacher_steepness)
        else:
            self.teacher_sorter = self.student_sorter
        self.tgt_emb_gc_sel = None

    def configure_optimizers(self):
        # Separate head params from backbone params
        head_params_named = []
        backbone_params_named = []
        block_numbers = []
        for name, param in self.model.named_parameters():
            if name.startswith("projection_head"):
                head_params_named.append((name, param))
            else:
                # add the general parameters of the backbone
                backbone_params_named.append((name, param))
                # and extract the block id/number
                reg_exp = re.search('.*blocks[.]([0-9]+)[.].*',name, re.IGNORECASE)
                block_number = None
                if reg_exp:
                    block_number = int(reg_exp.group(1))
                block_numbers.append(block_number)
        
        # filter out blocks not needed
        if self.trainable_blocks is not None:
            block_numbers_int = sorted(list(set([i for i in block_numbers if i is not None])))
            if type(self.trainable_blocks) == int:
                # Case 1: if a negative number, i.e. `-n` is provided then take the last n blocks
                # Case 2: if a positive number, i.e. `n` is provided then take the all the blocks from the nth and onwards blocks
                trainable_block_nums = block_numbers_int[self.trainable_blocks:]
            else:
                # Case 3: if a list is provided then select only the blocks from the list
                trainable_block_nums = self.trainable_blocks 
            trainable_block_nums = set(trainable_block_nums)
            
            # New backbone_params_named
            backbone_params_named_new = []
        
            for (name, param), bid in zip(backbone_params_named, block_numbers):
                if bid in trainable_block_nums:
                    backbone_params_named_new.append((name, param))
                else:
                    param.requires_grad_(False)
            backbone_params_named = backbone_params_named_new

        # Prepare param groups. Exclude norm and bias from weight decay if flag set.
        if self.exclude_norm_bias:
            backbone_params = self.exclude_from_wt_decay(backbone_params_named,
                                                         weight_decay=self.weight_decay,
                                                         lr=self.lr_backbone)
            head_params = self.exclude_from_wt_decay(head_params_named,
                                                     weight_decay=self.weight_decay,
                                                     lr=self.lr_heads)
            params = backbone_params + head_params
        else:
            backbone_params = [param for _, param in backbone_params_named]
            head_params = [param for _, param in head_params_named]
            params = [{'params': backbone_params, 'lr': self.lr_backbone},
                      {'params': head_params, 'lr': self.lr_heads}]

        # Init optimizer and lr schedule
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optim} not supported')
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_iters_per_epoch * self.max_epochs,
                                      eta_min=self.final_lr)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @staticmethod
    def exclude_from_wt_decay(named_params: Iterator[Tuple[str, nn.Parameter]], weight_decay: float, lr: float):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            # do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                excluded_params.append(param)
            else:
                params.append(param)
        return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
                {'params': excluded_params, 'weight_decay': 0., 'lr': lr}]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
                       optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None,
                       on_tpu: bool = None, using_native_amp: bool = None, using_lbfgs: bool = None,):

        if self.grad_norm_clipping is not None:
            params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                # do not regularize biases nor Norm parameters
                if name.endswith(".bias") or len(param.shape) == 1:
                    pass
                else:
                    params.append(param)
            
            torch.nn.utils.clip_grad_norm_(params, self.grad_norm_clipping)

        # Step weight decay schedule
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 2:
                param_group["weight_decay"] = self.wd_schedule[self.trainer.global_step]

        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, optimizer_idx)
        optimizer.step(closure=optimizer_closure)

    def shared_step(self, batch: Tuple[List[torch.Tensor], Dict]) -> float:
        inputs, bboxes = batch

        # 1. multi-res forward passes
        last_self_attention = True
        if self.loss_mask == "all":
            last_self_attention = False
        bs = inputs[0].size(0)
        if self.use_teacher:
            res_forward_teacher = self.teacher(inputs[:2], last_self_attention=last_self_attention)
        else:
            res_forward_teacher = self.model(inputs[:2], last_self_attention=last_self_attention)

        res_forward_student = self.model(inputs)

        if self.loss_mask == "all":
            teacher_gc_spatial_emb = res_forward_teacher[0]
        else:
            teacher_gc_spatial_emb, teacher_gc_attn = res_forward_teacher

        
        student_spatial_emb = res_forward_student[0]

        # 3. calculate gc and lc resolutions. Split student output in gc and lc embeddings
        gc_res_w = inputs[0].size(2) / self.patch_size
        gc_res_h = inputs[0].size(3) / self.patch_size
        assert gc_res_w.is_integer() and gc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        assert gc_res_w == gc_res_h, f"Only supporting square images not {inputs[0].size(2)}x{inputs[0].size(3)}"
        gc_spatial_res = int(gc_res_w)
        
        gc_student_spatial_emb, lc_student_spatial_emb = \
            student_spatial_emb[:bs * self.nmb_crops[0] * gc_spatial_res ** 2], \
            student_spatial_emb[bs * self.nmb_crops[0] * gc_spatial_res ** 2:]

        attn_hard = None
        if self.loss_mask != "all":
            # Merge attention heads and threshold attentions
            attn_smooth = sum(teacher_gc_attn[:, i] * 1 / teacher_gc_attn.size(1) for i
                              in range(teacher_gc_attn.size(1)))
            attn_smooth = attn_smooth.reshape(bs * self.nmb_crops[0], 1, gc_spatial_res, gc_spatial_res)
            attn_hard = process_attentions(attn_smooth, gc_spatial_res, threshold=0.6, blur_sigma=0.6)
            if self.loss_mask == 'bg':
                attn_hard = torch.abs(attn_hard - 1) # invert 1-0 mask if we want to train on bg tokens

        # compute lc spatial res
        lc_spatial_res = np.sqrt(lc_student_spatial_emb.size(0) / (self.nmb_crops[-1] * bs))
        if self.is_queue_usable:
            self.update_queue(teacher_gc_spatial_emb, bs, gc_spatial_res, lc_spatial_res, attn_hard) 
        # Calculate loss
        knn_loss = self.knn_loss(teacher_gc_spatial_emb, gc_student_spatial_emb,
                                         lc_student_spatial_emb, bboxes, bs, gc_spatial_res,
                                         lc_spatial_res, attn_hard=attn_hard)
        
        return knn_loss

    def compute_knn_perm_mat(self, spatial_emb_src: torch.Tensor, spatial_emb_tgt: torch.Tensor, sorter)-> float:
        bs, d, ps, ps2 = spatial_emb_src.shape
        assert ps == ps2

        src = spatial_emb_src.permute(0,2,3,1).reshape(bs, ps*ps, d) # (bs, ps*ps, d)
        tgt = spatial_emb_tgt.permute(0,2,3,1).reshape(bs, ps*ps, d) # (bs, ps*ps, d)
        if self.sim_dist == 'euclidean':
            dist = torch.cdist(src, tgt, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary') # (bs, ps*ps, ps*ps)
        elif self.sim_dist == 'cosine':
            # compute cosine similarity
            dist = torch.einsum('bhc,bwc->bhw', F.normalize(src, dim=-1), F.normalize(tgt, dim=-1))
        # dist = torch.norm(src.unsqueeze(2) - tgt.unsqueeze(1), dim=-1, p=2.0)
        dist = dist.reshape(-1, dist.shape[-1]) # (bs*ps*ps, ps*ps)
        
        _, perm_mat_teacher = sorter(dist) # (bs*ps*ps, ps*ps, ps*ps)
        # assert torch.all(torch.sum(perm_mat_teacher, dim=-1) == 1)
        return perm_mat_teacher.reshape(bs, ps, ps, ps*ps, ps*ps).permute(0, 4, 3, 1, 2) # (bs, ps*ps, ps*ps, ps, ps)


    def compute_knn_subloss(self, q, p, thresholded_mask, crop_idx):
        """
        Computes knn loss for a single crop pair.
        Args:
            q: query, shape (bs, C, L, H, W)
            p: target, shape (bs, C, L, H, W)
            attn_hard: attn mask, shape (bs, H, W)
        """
        if thresholded_mask is not None:
            mask = thresholded_mask[crop_idx::np.sum(self.nmb_crops)].squeeze().float().unsqueeze(1)
            if torch.sum(mask).item()!=0:
                # print('BEFORE mask.shape',mask.shape)
                mask = torch.repeat_interleave(mask, q.shape[2], dim=1)
                return -torch.sum(torch.sum(q * torch.log(p), dim=1) * mask) / torch.sum(mask)
            else:
                return 0.0
        else:
            # otherwise apply loss on all spatial tokens.
            return -torch.mean(torch.sum(q * torch.log(p), dim=1))

    def knn_loss(self, gc_teacher_emb: torch.Tensor, gc_student_emb: torch.Tensor,
                     lc_student_emb: torch.Tensor,  bboxes: Dict, bs: int, gc_spatial_res: int, 
                     lc_spatial_res, attn_hard: torch.Tensor = None, symmetric=False) -> float:
        assert lc_spatial_res.is_integer(), "spatial crops should have same x and y dim"
        lc_spatial_res = int(lc_spatial_res)

        # knn loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            # Select spatial cluster preds for global crop with crop_id
            emb_gc = gc_teacher_emb[bs * gc_spatial_res ** 2 * crop_id:bs * gc_spatial_res ** 2 * (crop_id + 1)] # (bs*ps*ps, 256)

            # Roi align of spatial outputs
            emb_gc_reshaped = emb_gc.reshape(bs, gc_spatial_res, gc_spatial_res, -1).permute(0, 3, 1, 2)
            downsampled_current_crop_boxes = torch.unbind(bboxes["gc"][:, crop_id] / self.patch_size)
            aligned_emb_gc = roi_align(emb_gc_reshaped, downsampled_current_crop_boxes,
                                              self.roi_align_kernel_size, aligned=True)  # (bs * num_crops, 7, 7, 2048)
            thresholded_mask = None
            if attn_hard is not None:
                # Roi align mask
                gc_hard_mask = attn_hard[bs * crop_id: bs * (crop_id+1)]  # select attn for crop_id
                aligned_mask = roi_align(gc_hard_mask, downsampled_current_crop_boxes, self.roi_align_kernel_size,
                                         aligned=True)
                thresholded_mask = (aligned_mask >= 1.0)  # Make mask 1-0

            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                if v in self.crops_for_assign:
                    # Code prediction from other global crop
                    out = gc_student_emb[bs * gc_spatial_res ** 2 * v:bs * gc_spatial_res ** 2 * (v + 1)]
                    spatial_res = gc_spatial_res
                else:
                    # Code prediction from local crop
                    lc_index = v - self.nmb_crops[0]
                    out = lc_student_emb[bs * lc_spatial_res**2 * lc_index:bs * lc_spatial_res**2 * (lc_index + 1)]
                    spatial_res = lc_spatial_res

                aligned_out = roi_align(out.reshape(bs, spatial_res, spatial_res, -1).permute(0, 3, 1, 2),
                                        torch.unbind(bboxes["all"][:, v, crop_id].unsqueeze(1) / self.patch_size),
                                        self.roi_align_kernel_size,
                                        aligned=True)
                aligned_emb_gc_sel = aligned_emb_gc[v::np.sum(self.nmb_crops)]
                bsl, d, ps1, ps2 = aligned_emb_gc_sel.shape
                num_elements = bsl * ps1 * ps2

                if self.is_queue_usable and self.entries_in_queue > 0:
                    # use a queue for selecting randomly reference patches from the last seen patches
                    idces = torch.randperm(self.queue[i].shape[0], device=self.device)[:num_elements]
                    idces = torch.remainder(idces, self.entries_in_queue)
                    tgt_emb_gc_sel = self.queue[i][idces].reshape(bsl, ps1, ps2, d).permute(0,3,1,2)
                else:
                    # otherwise select from the current batch
                    if thresholded_mask is not None:
                        # select patches after filtering of fg/bg patches
                        tmp_mask = thresholded_mask[bs * crop_id: bs * (crop_id+1)]
                        filtered_emg_gc = aligned_emb_gc_sel.permute(0,2,3,1).flatten(0,2)[tmp_mask.flatten()]
                    else:
                        # select from any patches
                        filtered_emg_gc = aligned_emb_gc_sel.permute(0,2,3,1).flatten(0,2)
                    if filtered_emg_gc.shape[0] == 0:
                        continue
                    repeats = round((bsl*ps1*ps2) / filtered_emg_gc.shape[0])+1
                    tgt_emb_gc_sel = filtered_emg_gc.repeat(repeats, 1)[:bsl*ps1*ps2]
                    idces = torch.randperm(tgt_emb_gc_sel.shape[0], device=self.device)
                    tgt_emb_gc_sel = tgt_emb_gc_sel[idces].reshape(bsl, ps1, ps2, d).permute(0,3,1,2)

                aligned_p = self.compute_knn_perm_mat(aligned_out, tgt_emb_gc_sel, sorter=self.student_sorter) # shape (bs, ps*ps, ps*ps, ps, ps)
                with torch.no_grad():
                    aligned_q = self.compute_knn_perm_mat(aligned_emb_gc_sel, tgt_emb_gc_sel, sorter=self.teacher_sorter) # shape (bs, ps*ps, ps*ps, ps, ps)
                
                subloss += self.compute_knn_subloss(aligned_q, aligned_p, thresholded_mask, v)

            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss


    def update_queue(self, gc_teacher_emb: torch.Tensor, bs:int, gc_spatial_res: int, lc_spatial_res: int, attn_hard: torch.Tensor = None) -> float:
        assert lc_spatial_res.is_integer(), "spatial crops should have same x and y dim"
        lc_spatial_res = int(lc_spatial_res)
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True

                    # Add spatial embeddings to queue
                    # Use attention to determine number of foreground embeddings to be stored
                    emb_gc = gc_teacher_emb[bs * gc_spatial_res ** 2 * crop_id:bs * gc_spatial_res ** 2 * (crop_id + 1)]
                    if attn_hard is not None:
                        # only add fg embeddings to queue
                        flat_mask = attn_hard.permute(0, 2, 3, 1).flatten().bool()
                        gc_fg_mask = flat_mask[bs * gc_spatial_res**2 * crop_id: bs * gc_spatial_res**2 * (crop_id+1)]
                        emb_gc = emb_gc[gc_fg_mask]
                    num_vectors_to_store = min(bs * 10, self.queue_length // self.gpus)
                    self.entries_in_queue = min((self.entries_in_queue + num_vectors_to_store), self.queue_length)
                    idx = torch.randperm(emb_gc.size(0))[:num_vectors_to_store]
                    self.queue[i, num_vectors_to_store:] = self.queue[i, :-num_vectors_to_store].clone()
                    self.queue[i, :num_vectors_to_store] = emb_gc[idx]

    def training_step(self, batch: Tuple[List[torch.Tensor], Dict], batch_idx: int) -> float:
        if isinstance(batch[1], dict):
            loss = self.shared_step(batch)
            if loss==0.0:
                return None
        else:
            raise ValueError("No rrc boxes passed")

        if self.use_teacher:
            # EMA update for the teacher using the momentum_schedule
            with torch.no_grad():
                m = self.momentum_schedule[self.trainer.global_step]  # momentum parameter
                for param_q, param_k in zip(self.model.parameters(), self.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        self.log('lr_backbone', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)
        self.log('lr_heads', self.optimizers().param_groups[2]['lr'], on_step=True, on_epoch=False)
        self.log('weight_decay', self.optimizers().param_groups[0]['weight_decay'], on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Validate for self.val_iters. Constrained to only parts of the validation set as mIoU calculation
        # would otherwise take too long.
        if self.val_iters is None or batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, mask = batch

                # Process gt seg masks
                bs = imgs.size(0)
                assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
                gt = mask * 255
                if self.val_downsample_masks:
                    size_masks = 100
                    gt = nn.functional.interpolate(gt, size=(size_masks, size_masks), mode='nearest')
                valid = (gt != 255)  # mask to remove object boundary class

                # Get backbone embeddings
                backbone_embeddings = self.model.forward_backbone(imgs)[:, 1:]

                # store embeddings, valid masks and gt for clustering after validation end
                res_w = int(np.sqrt(backbone_embeddings.size(1)))
                backbone_embeddings = backbone_embeddings.permute(0, 2, 1).reshape(bs, self.model.embed_dim,
                                                                                   res_w, res_w)
                self.preds_miou_layer4.update(valid, backbone_embeddings, gt)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # Trigger computations for rank 0 process
        res_kmeans = self.preds_miou_layer4.compute(self.trainer.is_global_zero)
        self.preds_miou_layer4.reset()
        if res_kmeans is not None:  # res_kmeans is none for all processes with rank != 0
            for k, name, res_k in res_kmeans:
                miou_kmeans, tp, fp, fn, _, matched_bg = res_k
                self.print(miou_kmeans)
                self.logger.experiment.log_metric(f'K={name}_miou_layer4', round(miou_kmeans, 8))
                # Log precision and recall values for each class
                for i, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
                    class_name = self.trainer.datamodule.class_id_to_name(i)
                    self.logger.experiment.log_metric(f'K={name}_{class_name}_precision',
                                                      round(tp_class / max(tp_class + fp_class, 1e-8), 8))
                    self.logger.experiment.log_metric(f'K={name}_{class_name}_recall',
                                                      round(tp_class / max(tp_class + fn_class, 1e-8), 8))
                if k > self.num_classes:
                    # Log percentage of clusters assigned to background class
                    self.logger.experiment.log_metric(f'K={name}-percentage-bg-cluster', round(matched_bg, 8))
