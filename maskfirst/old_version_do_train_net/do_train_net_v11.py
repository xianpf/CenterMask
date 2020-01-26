# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse, logging, time, datetime
import os
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
# from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # print('iteration:{}, losses:{}'.format(iteration, losses))
        if losses > 1e20 or torch.isnan(losses):
            import pdb; pdb.set_trace()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        if not losses.requires_grad:
            import pdb; pdb.set_trace()
        losses.backward()
        # try:
        #     losses.backward()
        # except:
        #     import pdb; pdb.set_trace()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        # if iteration % 100 == 0:
        #     # import pdb; pdb.set_trace()
        #     run_test(cfg, model, distributed=False, test_epoch=iteration)
        #     model.train()
        #     process_img()
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            run_test(cfg, model, distributed=False, test_epoch=iteration)
            model.train()
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.backbone import fpn as fpn_module
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import FCOSModule
from maskrcnn_benchmark.structures.bounding_box import BoxList

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class res_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(res_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.match_short = in_ch != out_ch
        if self.match_short:
            self.short_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False)
            self.short_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.match_short:
            identity = self.short_conv(identity)
            identity = self.short_bn(identity)

        x += identity
        x = F.relu_(x)
        return x

# class init_pyramid_conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(init_pyramid_conv, self).__init__()
#         self.resconv1 = res_conv(256+1, 256)
#         self.resconv2 = res_conv(256, 256)


# v11 feature: big update: no feature pyramid
class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        self.r50 = resnet.ResNet(cfg)
        self.res_layer_5 = nn.Sequential(
            resnet.Bottleneck(2048, 256, 256, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 512, 1, True, 1, 1, nn.BatchNorm2d),
        )
        self.res_layer_6 = nn.Sequential(
            resnet.Bottleneck(512, 256, 256, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )
        self.init_pyramid = nn.Sequential(
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.conv_6 = resnet.Bottleneck(256, 256, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.out_6 = nn.Conv2d(256, 2, 1)

        self.chn256_6 = nn.Conv2d(256+1, 256, 1)
        self.chn256_5 = nn.Conv2d(256+1, 256, 1)
        self.chn256_4 = nn.Conv2d(256+1, 256, 1)
        self.chn256_3 = nn.Conv2d(256+1, 256, 1)

        self.test0 = {}
        self.test1 = {}
        self.loss_evaluators = []
        # self.fcos = FCOSModule(cfg, 256)
        # self.p7p6conv = nn.Conv2d(256, 1, 1)

        # self.mask_conv_0 = nn.Conv2d(256+1, 2, 7, padding=3)
        # self.mask_conv_0_0 = nn.Conv2d(256+1, 256, 3, padding=3)
        # self.mask_conv_0_1 = nn.Conv2d(256, 256, 3, padding=3)
        # self.mask_conv_0_2 = nn.Conv2d(256, 1, 3, padding=3)
        # self.mask_conv_0 = double_conv(256+1, 2)
        # self.mask_conv_0 = res_conv(256+1, 2)
        self.mask_conv_0 = nn.Sequential(
            res_conv(256+1, 256),
            res_conv(256, 256),
            nn.Conv2d(256, 2, 1)
        )

        self.mask_conv_1 = nn.Conv2d(256+1, 1, 3, padding=1)
        self.mask_conv_2 = nn.Conv2d(256+1, 1, 3, padding=1)
        self.mask_conv_3 = nn.Conv2d(256+1, 1, 3, padding=1)
        self.mask_conv_4 = nn.Conv2d(256+1, 1, 3, padding=1)

        self.mask_conv_0_bn_0 = nn.BatchNorm2d(2)
        self.cs_criteron = nn.CrossEntropyLoss()

        self.mask_convs = [self.mask_conv_0, self.mask_conv_1, self.mask_conv_2, self.mask_conv_3, self.mask_conv_4]
        self.low_thresh = 0.2

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_target(self, img_tensor_shape, device, target=None):
        target_ori_mask = target.get_field('masks').get_mask_tensor().unsqueeze(0).to(device)

        target_shape = (1, target_ori_mask.shape[-3]) + img_tensor_shape
        target_mask_pad_to_img = target_ori_mask.new(*target_shape).zero_()
        target_mask_pad_to_img[:,:,:target.size[1], :target.size[0]] = target_ori_mask
        
        target_levels = {}
        target_levels[0] = target_mask_pad_to_img
        level_shape = ((img_tensor_shape[0]+1)//2, (img_tensor_shape[1]+1)//2)
        target_levels[1] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[2] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[3] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[4] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[5] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[6] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)
        level_shape = ((level_shape[0]+1)//2, (level_shape[1]+1)//2)
        target_levels[7] = F.interpolate(target_mask_pad_to_img.float(), level_shape, mode='nearest').type(target_mask_pad_to_img.dtype)

        # import pdb; pdb.set_trace()
        # print([t.shape for t in target_levels.values()])
        return target_levels
   
    def compute_mask_sigmoid(self, level, feature, pyramids, is_init=False):
        for pyramid in pyramids:
            # import pdb; pdb.set_trace()
            feature_gaussian_mask = pyramid.get_feature_gaussian_mask(level, feature.shape[-2:]).to(feature.device)
            masked_feature = feature * feature_gaussian_mask
            if is_init:
                last_mask = pyramid.get_feature_gaussian_mask(level, feature.shape[-2:])[None, None,:,:].to(feature.device)
            else:
                last_mask = pyramid.get_mask(level-1)
                up_size = tuple(feature.shape[-2:])
                last_mask = F.interpolate(last_mask, up_size, mode='bilinear', align_corners=False)

            # 把cat 改成 高斯mask相乘， ablation study
            conv_in = torch.cat((feature, last_mask), dim=1)
            # conv_in = torch.cat((masked_feature, last_mask), dim=1)
            # TODO: self.mask_convs[level-pyramid.init_level]
            # mask = torch.sigmoid(self.mask_convs[level](conv_in))
            mask_logits = self.mask_convs[level](conv_in)
            # mask = torch.sigmoid(mask_logits)
            mask = torch.sigmoid(self.bn_0(mask_logits))
            pyramid.set_mask(level, mask)

    def compute_mask(self, level, feature, pyramids, is_init=False):
        for j, pyramid in enumerate(pyramids):
            feature_gaussian_mask = pyramid.get_feature_gaussian_mask(level, feature.shape[-2:]).to(feature.device)
            # masked_feature = feature * feature_gaussian_mask
            # masked_feature_plus = feature + feature_gaussian_mask
            # self.test0[j] = feature_gaussian_mask
            # self.test1[j] = masked_feature_plus
            if is_init:
                last_mask = pyramid.get_feature_gaussian_mask(level, feature.shape[-2:])[None, None,:,:].to(feature.device)
                conv_in = torch.cat((feature, last_mask), dim=1)
                out = self.chn256_6(conv_in)
                out = self.init_pyramid(out)
                mask_logits = self.conv_6(out)
                pyramid.set_mask_logits(level, mask_logits)
                mask = self.out_6(mask_logits)
                pyramid.set_mask(level, mask)
            else:
                import pdb; pdb.set_trace()
                last_mask = pyramid.get_mask(level-1)
                up_size = tuple(feature.shape[-2:])
                last_mask = F.interpolate(last_mask, up_size, mode='bilinear', align_corners=False)

            # # 把cat 改成 高斯mask相乘， ablation study
            # # conv_in = torch.cat((feature, last_mask), dim=1)
            # conv_in = torch.cat((masked_feature, last_mask), dim=1)
            # # conv_in = torch.cat((masked_feature_plus, last_mask), dim=1)
            # # TODO: self.mask_convs[level-pyramid.init_level]
            # # mask = torch.sigmoid(self.mask_convs[level](conv_in))
            # mask_logits = self.mask_convs[level](conv_in)
            # # mask = torch.sigmoid(mask_logits)
            # # mask = torch.sigmoid(self.bn_0(mask_logits))
            # # pyramid.set_mask(level, mask)
            # pyramid.set_mask(level, mask_logits)

    def compute_loss(self, level, pyramids, target_levels):
        losses = []
        covered_idx = []
        for pyramid in pyramids:
            mask = pyramid.get_mask(level)
            target = target_levels[7-level]
            # import pdb; pdb.set_trace()
            target_idx = target[0, :, pyramid.pos[0], pyramid.pos[1]].nonzero()

            if len(target_idx):
                if len(target_idx) > 1:
                    target_idx = target[0, target_idx].view(len(target_idx),\
                        -1).sum(1).argmin().unsqueeze(0).unsqueeze(0)
                target_mask = target[:, [target_idx]]
                pyramid.bind_target(target_idx)
                covered_idx.append(target_idx)
                loss = (mask - target_mask.float()).abs().mean()
            else:
                # 问题出在这里：要element product
                loss = (target.float().sum(dim=1).unsqueeze(1) * mask).mean()
            losses.append(loss)
        # TODO: 检查未被追踪的target_idx
        # if not len(covered_idx):
        #     import pdb; pdb.set_trace()
        covered_idx = torch.cat(covered_idx).unique() if len(covered_idx) else []
        missed_target = torch.nonzero(target[0].sum(dim=0).type(torch.bool)^target[0, covered_idx].sum(dim=0).type(torch.bool))
        if len(missed_target):
            # print('\n\n\n\nxxxxxxxxxxxxx\n', covered_idx, target.shape)
            # import pdb; pdb.set_trace()
            new_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
            loss_miss = new_masks[0,:,missed_target[:,0], missed_target[:,1]].mean()
            losses.append(loss_miss)
        
        resloss = sum(loss for loss in losses) / len(losses)
        return resloss
        
    def compute_loss_target_only(self, level, pyramids, target_levels):
        losses = []
        covered_idx = []
        for pyramid in pyramids:
            mask = pyramid.get_mask(level)
            target = target_levels[7-level]
            # import pdb; pdb.set_trace()
            target_idx = target[0, :, pyramid.pos[0], pyramid.pos[1]].nonzero()

            if len(target_idx):
                if len(target_idx) > 1:
                    target_idx = target[0, target_idx].view(len(target_idx),\
                        -1).sum(1).argmin().unsqueeze(0).unsqueeze(0)
                target_mask = target[:, [target_idx]]
                import pdb; pdb.set_trace()
                pyramid.bind_target(target_idx)
                covered_idx.append(target_idx)
                loss = (mask - target_mask.float()).abs().mean()
                losses.append(loss)
            # else:
                # 问题出在这里：要element product
                # loss = (target.float().sum(dim=1).unsqueeze(1) * mask).mean()
            # losses.append(loss)
        # TODO: 检查未被追踪的target_idx
        # if not len(covered_idx):
        #     import pdb; pdb.set_trace()
        covered_idx = torch.cat(covered_idx).unique() if len(covered_idx) else []
        missed_target = torch.nonzero(target[0].sum(dim=0).type(torch.bool)^target[0, covered_idx].sum(dim=0).type(torch.bool))
        if len(missed_target):
            # print('\n\n\n\nxxxxxxxxxxxxx\n', covered_idx, target.shape)
            import pdb; pdb.set_trace()
            new_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
            loss_miss = new_masks[0,:,missed_target[:,0], missed_target[:,1]].mean()
            losses.append(loss_miss)
        
        # if len(losses) <= 0:
        #     import pdb; pdb.set_trace()
        resloss = sum(loss for loss in losses) / len(losses) if len(losses) else torch.tensor(0.0, device=mask.device)
        return resloss
        
    def compute_loss_softmax(self, level, pyramids, target_levels):
        losses = []
        covered_idx = []
        for pyramid in pyramids:
            mask_logits = pyramid.get_mask(level)
            target = target_levels[7-level]
            # import pdb; pdb.set_trace()
            target_idx = target[0, :, pyramid.pos[0], pyramid.pos[1]].nonzero()

            if len(target_idx):
                if len(target_idx) > 1:
                    target_idx = target[0, target_idx].view(len(target_idx),\
                        -1).sum(1).argmin().unsqueeze(0).unsqueeze(0)
                target_mask = target[:, [target_idx]]
                pyramid.bind_target(target_idx)
                covered_idx.append(target_idx)

                # import pdb; pdb.set_trace()
                loss_cs = self.cs_criteron(mask_logits, target_mask.squeeze(1).long())
                losses.append(loss_cs)

                if loss_cs > 1e20 or torch.isnan(loss_cs):
                    import pdb; pdb.set_trace()

                # loss = (mask - target_mask.float()).abs().mean()
                # losses.append(loss)

            # else:
                # 问题出在这里：要element product
                # loss = (target.float().sum(dim=1).unsqueeze(1) * mask).mean()
            # losses.append(loss)
        # TODO: 检查未被追踪的target_idx
        # if not len(covered_idx):
        #     import pdb; pdb.set_trace()
        covered_idx = torch.cat(covered_idx).unique() if len(covered_idx) else []
        missed_target = torch.nonzero(target[0].sum(dim=0).type(torch.bool)^target[0, covered_idx].sum(dim=0).type(torch.bool))
        # if len(missed_target):
        if False and len(missed_target):
            # print('\n\n\n\nxxxxxxxxxxxxx\n', covered_idx, target.shape)
            import pdb; pdb.set_trace()
            new_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
            loss_miss = new_masks[0,:,missed_target[:,0], missed_target[:,1]].mean()
            losses.append(loss_miss)
        
        # if len(losses) <= 0:
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        if len(losses):
            resloss = sum(loss for loss in losses) / len(losses) 
        else:
            # import pdb; pdb.set_trace()
            # resloss = torch.tensor(0.0, device=target.device)
            # resloss = torch.tensor(0.0, device=target.device, requires_grad=False)
            resloss = mask_logits.sum()*0
        return resloss
        
    # V10 feature: softmax cross entropy
    def forward(self, image, targets=None):
        x_img = image.tensors
        xs_r50 = self.r50(x_img)

        # import pdb; pdb.set_trace()
        xs_r50.append(self.res_layer_5(xs_r50[-1]))
        xs_r50.append(self.res_layer_6(xs_r50[-1]))


        # fs_fpn = self.fpn(xs_r50)
        N, _, img_size_h, img_size_w = x_img.shape
        device = x_img.device
        level_sizes = [tuple(f.shape[-2:]) for f in xs_r50[::-1]]
        # level_sizes = [tuple(f.shape[-2:]) for f in xs_r50]

        losses = {}
        losses_0 = []
        losses_1 = []
        losses_2 = []
        losses_3 = []
        losses_4 = []
        test_masks = []
        for i in range(N):
            if self.training:
                target_levels = self._init_target((img_size_h, img_size_w ), device, targets[i])
            
            curr_level = 0
            x_curr = xs_r50[-1]
            init_pos = torch.nonzero(torch.ones_like(x_curr[0][0]))
            inst_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in init_pos]
            self.compute_mask(curr_level, x_curr[[i]], inst_pyramids, True)
            if self.training:
                loss_0 = self.compute_loss_softmax(0, inst_pyramids, target_levels)
                losses_0.append(loss_0)
            





            # curr_level = 0
            # x_curr = fs_fpn[::-1][curr_level]
            # init_pos = torch.nonzero(torch.ones_like(x_curr[0][0]))
            # inst_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in init_pos]
            # self.compute_mask(curr_level, x_curr[[i]], inst_pyramids, True)
            # if self.training:
            #     losses_0.append(self.compute_loss_softmax(0, inst_pyramids, target_levels))

            # curr_level = 1
            # x_curr = fs_fpn[::-1][curr_level]
            # # import pdb; pdb.set_trace()
            # self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
            # new_masks = torch.cat([i_p.get_mask(curr_level) for i_p in inst_pyramids], dim=1)
            # new_pos = torch.nonzero(new_masks[0].max(dim=0)[0] < self.low_thresh)
            # new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
            # self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
            # inst_pyramids += new_pyramids
            # if self.training:
            #     losses_1.append(self.compute_loss_target_only(1, inst_pyramids, target_levels))

            # curr_level = 2
            # x_curr = fs_fpn[::-1][curr_level]
            # # import pdb; pdb.set_trace()
            # self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
            # new_masks = torch.cat([i_p.get_mask(curr_level) for i_p in inst_pyramids], dim=1)
            # new_pos = torch.nonzero(new_masks[0].max(dim=0)[0] < self.low_thresh)
            # new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
            # self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
            # inst_pyramids += new_pyramids

            # if self.training:
            #     losses_2.append(self.compute_loss_target_only(2, inst_pyramids, target_levels))
            
            if not self.training:
                test_masks.append(inst_pyramids)

            
        # import pdb; pdb.set_trace()
        losses['level_0']= sum(loss for loss in losses_0)
        losses['level_1']= sum(loss for loss in losses_1)
        losses['level_2']= sum(loss for loss in losses_2)
        losses['level_3']= sum(loss for loss in losses_3)
        losses['level_4']= sum(loss for loss in losses_4)
        return losses if self.training else test_masks

class InstancePyramid():
    def __init__(self, pos, init_pub_level, level_sizes):
        self.init_level = init_pub_level
        self.level_sizes = level_sizes
        self.pos = pos
        self.masks = {}
        self.mask_logits = {}
        self.target_idx = None
        self.is_alive = True
        # torch.tensor(800.0/2**(7-self.init_level)).ceil().long().item()
        self.feature_scales = [7, 13, 25, 50, 100, 200]
        self.gaussian_masks = self.generate_gaussian_masks()

    def set_mask(self, pub_level, mask):
        self.masks[pub_level - self.init_level] = mask

    def get_mask(self, pub_level):
        return self.masks[pub_level - self.init_level]


    def set_mask_logits(self, pub_level, mask_logits):
        self.mask_logits[pub_level - self.init_level] = mask_logits

    def get_mask_logits(self, pub_level):
        return self.mask_logits[pub_level - self.init_level]

    def bind_target(self, idx):
        self.target_idx = idx

    def compute_loss(self, target, pub_level):
        import pdb; pdb.set_trace()


    def get_root_level_pos(self, pub_level):
        init_size = self.level_sizes[self.init_level]
        req_size = self.level_sizes[pub_level]

        h = (self.pos[0].float() / init_size[0] * req_size[0]).round().long()
        w = (self.pos[1].float() / init_size[1] * req_size[1]).round().long()

        return (h.item(), w.item())

    def get_root_response(self, pub_level):
        init_size = self.level_sizes[self.init_level]
        req_size = self.level_sizes[pub_level]
        # h1 = (self.pos[0].float() / init_size[0] * req_size[0]).floor()
        # h2 = ((self.pos[0].float()+1) / init_size[0] * req_size[0]).ceil()
        # w1 = (self.pos[1].float() / init_size[1] * req_size[1]).floor()
        # w2 = ((self.pos[1].float()+1) / init_size[1] * req_size[1]).ceil()

        h = (self.pos[0].float() / init_size[0] * req_size[0]).round().long()
        w = (self.pos[1].float() / init_size[1] * req_size[1]).round().long()

        points = self.masks[pub_level - self.init_level][0,0,h, w]

        return points

    def generate_gaussian_masks(self):
        # torch.tensor(800.0/2**(7-self.init_level)).ceil().long().item()
        # feature_scales = [7, 13, 25, 50, 100, 200]
        gaussian_masks = []
        for i in range(len(self.feature_scales)):
            f_scale = self.feature_scales[i]
            xs = torch.arange(f_scale*4)
            ys = torch.arange(f_scale*4).view(-1,1)
            gaussian_masks.append((-4*(torch.tensor(2.0, requires_grad=False)).log()*((xs.float()-f_scale*\
                2+1)**2+(ys.float()-f_scale*2+1)**2)/f_scale**2).exp())

        return gaussian_masks

    def get_feature_gaussian_mask(self, pub_level, feature_size):
        # gaussian_mask = self.gaussian_masks[pub_level - self.init_level]
        feature_size = tuple(feature_size)
        gaussian_mask = self.gaussian_masks[pub_level]
        level_pos = self.get_root_level_pos(pub_level)
        ctr = (self.feature_scales[pub_level]*2-1,)*2
        feature_g_mask = gaussian_mask[ctr[0]-level_pos[0]:ctr[0]-level_pos[0]+feature_size[0], \
            ctr[1]-level_pos[1]:ctr[1]-level_pos[1]+feature_size[1]]
        return feature_g_mask

def train(cfg, local_rank, distributed):
    # model = build_detection_model(cfg)
    model = MaskPyramids(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        # cfg, model, optimizer, scheduler, output_dir, save_to_disk
        cfg, model.r50, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model

def get_neat_inference_result(coco_eval):
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = coco_eval.params
        # import pdb; pdb.set_trace()
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]   
            # [p.iouThrs * p.recThrs * p.catIds * p.areaRng * p.maxDets]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        summaryStr = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, summaryStr
    def _summarizeDets():
        stats = np.zeros((12,))
        summaryStrs = [None] * 12
        stats[0], summaryStrs[0] = _summarize(1)
        stats[1], summaryStrs[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2], summaryStrs[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3], summaryStrs[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4], summaryStrs[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5], summaryStrs[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6], summaryStrs[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7], summaryStrs[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8], summaryStrs[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9], summaryStrs[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10], summaryStrs[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11], summaryStrs[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats, summaryStrs
 
    stats, summaryStrs = _summarizeDets()
    return summaryStrs

def run_test(cfg, model, distributed, test_epoch=None):
    return 0
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        # import pdb; pdb.set_trace()
        summaryStrs = get_neat_inference_result(inference_result[2][0])
        # print('\n'.join(summaryStrs))
        summaryStrFinal = '\n'.join(summaryStrs)
        summaryStrFinal = '\n\nEpoch: ' + str(test_epoch) + '\n' + summaryStrFinal
        # with open(output_folder+'/summaryStrs.txt', 'w') as f_summaryStrs:
        with open(output_folder+'/summaryStrs.txt', 'a') as f_summaryStrs:
            f_summaryStrs.write(summaryStrFinal)

def process_img():
    import pdb; pdb.set_trace()
    import glob
    # datasets/coco/val2014
    imglist = glob(os.path.join(args.input, '*'))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="maskfirst/mask_first.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.merge_from_list(['OUTPUT_DIR', 'run/develop'])
    # cfg.merge_from_list(['MODEL.WEIGHT', 'run/try_2/model_0010000.pth'])
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'new_config.yml')
        with open(output_config_path, 'w') as f:
            f.write(cfg.dump())

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
