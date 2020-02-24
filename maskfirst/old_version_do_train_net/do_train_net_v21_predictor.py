# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse, logging, time, datetime
import os, cv2, glob
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
        # if losses > 1e20 or torch.isnan(losses):
        if torch.isnan(losses):
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
            logger.info(("# of instances: level 0: {}\t|level 1: {}\t|level 2: {}\t|"+\
                "level 3: {}\t|level 4: {}\t|level 5: {} | # of instances counts: {}").format(
                model.log_dict['pyr_num_l0'], model.log_dict['pyr_num_l1'], 
                model.log_dict['pyr_num_l2'], model.log_dict['pyr_num_l3'], 0, 0, model.log_dict['InstPyr_inst_count']))
            # logger.info("# of instances counts: {}".format(model.log_dict['InstPyr_inst_count']))
        # if iteration % 100 == 0:
        if iteration % 40 == 0:
            run_test(cfg, model, distributed=False, test_epoch=iteration)
            model.train()
        #     process_img()
        if iteration % checkpoint_period == 0:
            # checkpointer.save("model_{:07d}".format(iteration), **arguments)
            save_data = {}
            save_data["model"] = model.state_dict()
            if optimizer is not None:
                save_data["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                save_data["scheduler"] = scheduler.state_dict() 
            torch.save(save_data, cfg.OUTPUT_DIR+"/model_{:07d}.pth".format(iteration))
            run_test(cfg, model, distributed=False, test_epoch=iteration)
            model.train()
        if iteration == max_iter:
            # checkpointer.save("model_final", **arguments)
            save_data = {}
            save_data["model"] = model.state_dict()
            if optimizer is not None:
                save_data["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                save_data["scheduler"] = scheduler.state_dict() 
            torch.save(save_data, cfg.OUTPUT_DIR+"/model_final.pth")

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

class ResNet50(nn.Module):
    def __init__(self, cfg):
        super(ResNet50, self).__init__()
        self.stem = resnet.BaseStem(cfg, nn.BatchNorm2d)


        self.res_layer_1 = nn.Sequential(
            resnet.Bottleneck(64, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_2 = nn.Sequential(
            resnet.Bottleneck(256, 128, 512, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(512, 128, 512, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_3 = nn.Sequential(
            resnet.Bottleneck(512, 256, 1024, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(1024, 256, 1024, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(1024, 256, 1024, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(1024, 256, 1024, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(1024, 256, 1024, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(1024, 256, 1024, 1, True, 1, 1, nn.BatchNorm2d),
        )

        self.res_layer_4 = nn.Sequential(
            resnet.Bottleneck(1024, 512, 2048, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(2048, 512, 2048, 1, True, 1, 1, nn.BatchNorm2d),
            resnet.Bottleneck(2048, 512, 2048, 1, True, 1, 1, nn.BatchNorm2d),
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.res_layer_1(x)
        outputs.append(x)
        x = self.res_layer_2(x)
        outputs.append(x)
        x = self.res_layer_3(x)
        outputs.append(x)
        x = self.res_layer_4(x)
        outputs.append(x)
        
        return outputs

# v21 feature: class 共同基于mask logits 3,全连接class logits
class MaskPyramids(nn.Module):
    def __init__(self, cfg):
        super(MaskPyramids, self).__init__()
        self.cfg = cfg
        # self.r50 = resnet.ResNet(cfg)
        self.resnet50 = ResNet50(cfg)
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

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
        self.class_logits = nn.Sequential(
            resnet.Bottleneck(256, 64, 128, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(128, 64, 64, 1, True, 2, 1, nn.BatchNorm2d),
            resnet.Bottleneck(64, 32, 32, 1, True, 2, 1, nn.BatchNorm2d),
        )
        self.cls_score = nn.Linear(32*7*7, num_classes)

        self.conv_6 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_5 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_4 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_3 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_2 = resnet.Bottleneck(256, 64, 256, 1, True, 1, 1, nn.BatchNorm2d)
        self.conv_s = [self.conv_6, self.conv_5, self.conv_4, self.conv_3, self.conv_2]
        self.out_6 = nn.Conv2d(256, 1, 1)
        self.out_5 = nn.Conv2d(256, 1, 1)
        self.out_4 = nn.Conv2d(256, 1, 1)
        self.out_3 = nn.Conv2d(256, 1, 1)
        self.out_2 = nn.Conv2d(256, 1, 1)
        self.out_s = [self.out_6, self.out_5, self.out_4, self.out_3, self.out_2]
        self.bg_6 = nn.Conv2d(256, 1, 1)
        self.bg_5 = nn.Conv2d(512, 1, 1)
        self.bg_4 = nn.Conv2d(2048, 1, 1)
        self.bg_3 = nn.Conv2d(1024, 1, 1)
        self.bg_2 = nn.Conv2d(512, 1, 1)
        self.bg_s = [self.bg_6, self.bg_5, self.bg_4, self.bg_3, self.bg_2]

        self.chn256_6 = nn.Conv2d(256+1, 256, 1)
        self.chn256_5 = nn.Conv2d(512+1, 256, 1)
        self.chn256_4 = nn.Conv2d(2048+1, 256, 1)
        self.chn256_3 = nn.Conv2d(1024+1, 256, 1)
        self.chn256_2 = nn.Conv2d(512+1, 256, 1)
        self.chn256_s = [self.chn256_6, self.chn256_5, self.chn256_4, self.chn256_3, self.chn256_2]

        self.upmix256_5 = nn.Conv2d(256+512+1, 256, 1)
        self.upmix256_4 = nn.Conv2d(256+2048+1, 256, 1)
        self.upmix256_3 = nn.Conv2d(256+1024+1, 256, 1)
        self.upmix256_2 = nn.Conv2d(256+512+1, 256, 1)
        self.upmix256_s = [None, self.upmix256_5, self.upmix256_4, self.upmix256_3, self.upmix256_2]

        self.test0 = {}
        self.test1 = {}
        self.log_dict = {}
        self.loss_evaluators = []
        
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
        self.class_criteron = nn.CrossEntropyLoss()

        self.mask_convs = [self.mask_conv_0, self.mask_conv_1, self.mask_conv_2, self.mask_conv_3, self.mask_conv_4]
        self.low_thresh = 0.2
        self.cs_loss_factor = 1.0
        self.miss_loss_factor = 1.0
        self.class_loss_factor = 1.0

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

        target_levels['labels'] = target.get_field('labels')
        return target_levels
   
    def compute_class(self, pyramids, target_levels):
        class_losses = []
        for pyramid in pyramids:
            mask_logits_3 = pyramid.get_mask_logits(3)
            class_logits = self.class_logits(mask_logits_3)
            class_logits_7x7 = F.interpolate(class_logits, (7, 7))
            class_logits = self.cls_score(class_logits_7x7.view(class_logits_7x7.shape[0],-1))
            pyramid.class_logits = class_logits

            if pyramid.target_idx is not None:
                target_label = target_levels['labels'][[pyramid.target_idx]]
                loss_class = self.class_criteron(class_logits, target_label)
                class_losses.append(loss_class)

        class_loss = sum(loss for loss in class_losses) / len(class_losses)\
            if len(class_losses) else mask_logits_3.sum()*0

        return class_loss

    def compute_mask(self, level, feature, pyramids, is_init=False):
        x_shared_bg = self.bg_s[level](feature)
        for pyramid in pyramids:
            # feature_gaussian_mask = pyramid.get_feature_gaussian_mask(level, feature.shape[-2:]).to(feature.device)
            # feature_gaussian_mask = pyramid.get_feature_gaussian_mask(level, feature[0,0]).to(feature.device)
            # import pdb; pdb.set_trace()
            feature_gaussian_mask = F.interpolate(pyramid.init_gaussian_mask[None,None,...]\
                .to(feature.device), tuple(feature[0,0].shape))[0,0]
            if is_init:
                conv_in = torch.cat((feature, feature_gaussian_mask[None, None,:,:]), dim=1)
                chn256 = self.chn256_s[level](conv_in)
                x_init = self.init_pyramid(chn256)
                mask_logits = self.conv_s[level](x_init)
                pyramid.set_mask_logits(level, mask_logits)
                mask_1 = self.out_s[level](mask_logits)
                mask = torch.cat((x_shared_bg, mask_1), dim=1)
                pyramid.set_mask(level, mask)
                
                # # import pdb; pdb.set_trace()
                # class_logits = self.class_logits(chn256)
                # class_logits = F.adaptive_avg_pool2d(class_logits, (1,1)).squeeze(-1).squeeze(-1)
                # pyramid.class_logits = class_logits

            else:
                last_mask_logits = pyramid.get_mask_logits(level-1)
                up_size = tuple(feature.shape[-2:])
                last_mask_logits_up = F.interpolate(last_mask_logits, up_size, mode='bilinear', align_corners=False)

                conv_in = torch.cat((last_mask_logits_up, feature, feature_gaussian_mask[None, None,:,:]), dim=1)
                out = self.upmix256_s[level](conv_in)
                mask_logits = self.conv_s[level](out)
                pyramid.set_mask_logits(level, mask_logits)
                mask_1 = self.out_s[level](mask_logits)
                mask = torch.cat((x_shared_bg, mask_1), dim=1)
                pyramid.set_mask(level, mask)

            # if mask_logits.max() > 1e20 or torch.isnan(mask_logits.max()):
            if torch.isnan(mask_logits.max()):
                import pdb; pdb.set_trace()
        # return feature_gaussian_mask
 
    def compute_loss(self, level, pyramids, target_levels, target_support_pyramids):
        # TODO: multi target cross entropy loss
        losses = []
        miss_losses = []
        # class_losses = []
        # covered_idx = []
        for pyramid in pyramids:
            mask_logits = pyramid.get_mask(level)
            if pyramid.target_idx is not None:
                target_mask = target_levels[7-level][0, [pyramid.target_idx]]
                loss_cs = self.cs_criteron(mask_logits, target_mask.squeeze(1).long())
                losses.append(loss_cs)

                # if loss_cs > 1e20 or torch.isnan(loss_cs):
                if torch.isnan(loss_cs):
                    import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                # if level == 3:
                #     import pdb; pdb.set_trace()
                #     target_label = target_levels['labels'][[pyramid.target_idx]]
                #     loss_class = self.class_criteron(pyramid.class_logits, target_label)
                #     class_losses.append(loss_class)

        # import pdb; pdb.set_trace()
        # TODO: 检查未被追踪的target_idx
        # TODO: 避免惩罚该target 指导的pyramid， 不存在这个问题。。。
        for i, t_match_list in  enumerate(target_support_pyramids):
            if not t_match_list:    # target 没有match到pyramid
                miss_target_map = target_levels[7-level][0, i]
                if miss_target_map.sum():
                    miss_pos = miss_target_map.nonzero()
                    # import pdb; pdb.set_trace()
                    all_masks = torch.cat([i_p.get_mask(level) for i_p in pyramids], dim=1)
                    loss_miss = all_masks[0,:,miss_pos[:,0], miss_pos[:,1]].mean()

        cs_loss = sum(loss for loss in losses)/len(losses)\
            if len(losses) else mask_logits.sum()*0
        miss_loss = sum(loss for loss in miss_losses) / len(miss_losses)\
            if len(miss_losses) else mask_logits.sum()*0
        # class_loss = sum(loss for loss in class_losses) / len(class_losses)\
        #     if len(class_losses) else mask_logits.sum()*0

        # resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor + \
        #     class_loss * self.class_loss_factor

        resloss = cs_loss * self.cs_loss_factor + miss_loss * self.miss_loss_factor
                    
        # resloss = (sum(loss for loss in losses)/len(losses)\
        #     if len(losses) else mask_logits.sum()*0) * self.cs_loss_factor + \
        #     (sum(loss for loss in miss_losses) / len(miss_losses)\
        #     if len(miss_losses) else mask_logits.sum()*0) * self.miss_loss_factor
        return resloss

    # 只能给新来的 pyramids match target， 旧的match 要保持连贯性
    def match_target(self, level, pyramids, target_levels, target_support_pyramids):
        for pyr in pyramids:
            target_idxs = target_levels[7-level][0, :, pyr.pos[0], pyr.pos[1]].nonzero()
            for i, target_idx in enumerate(target_idxs):
                target_idx_int = target_idx[0].item()
                target_support_pyramids[target_idx_int].append(pyr.idx)
                # 解决一个pixel 多个target 的问题， 核心：(小target优先)， 已分配的不改
                if pyr.target_idx is not None:
                    # print('target_idxs', target_idxs)
                    # print('pyr.target_idx:', pyr.target_idx)
                    target_map_last = target_levels[7-level][0, pyr.target_idx]
                    target_map_now = target_levels[7-level][0, target_idx_int]
                    # 重叠的上一个已经有其他pixel选项了， 让其让路
                    if len(target_support_pyramids[pyr.target_idx]) > 1:
                        target_support_pyramids[pyr.target_idx].remove(pyr.idx)
                        pyr.bind_target(target_idx_int)
                    elif len(target_support_pyramids[target_idx_int]) > 1:
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    elif (target_map_now == target_map_last).all():
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    elif target_map_now.sum() < target_map_last.sum():
                        target_support_pyramids[pyr.target_idx].remove(pyr.idx)
                        pyr.bind_target(target_idx_int)
                    elif target_map_now.sum() > target_map_last.sum():
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                    else:
                        target_support_pyramids[target_idx_int].remove(pyr.idx)
                        continue
                        # import pdb; pdb.set_trace()
                else:
                    pyr.bind_target(target_idx_int)

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        import pdb; pdb.set_trace()
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


    # def pyr_list_to_boxlist(self, pyramids):
    def pyr_list_to_boxlist_visual(self, pyramids, image, image_size):
        show_img = image.tensors[0].permute(1,2,0)
        show_img = (show_img - show_img.min())/(show_img.max() - show_img.min())
        show_img = show_img.detach().cpu().numpy()

        integrated_masks = torch.cat([pyramids[0].get_mask(3)[:, 0]]+\
            [pyr.get_mask(3)[:, 1] for pyr in pyramids], dim=0).unsqueeze(0)
        # import pdb; pdb.set_trace()
        masks_sum = [pyr.get_mask(3)[0].max(dim=0)[1].sum().item() for pyr in pyramids]
        valid_boxes = []
        valid_pyramids = []
        for pyr in pyramids:
            mask = pyr.get_mask(3)[0].max(dim=0)[1]
            if mask.sum():
                # import pdb; pdb.set_trace()
                ones = mask.nonzero()
                box = torch.cat((ones.min(dim=0)[0], ones.max(dim=0)[0]))
                valid_boxes.append(box)
                valid_pyramids.append(pyr)

        # import pdb; pdb.set_trace()
        boxes = torch.stack(valid_boxes, dim=0)
        if len(valid_boxes) == 0:
            import pdb; pdb.set_trace()

        class_logits = torch.cat([pyr.class_logits for pyr in valid_pyramids])
        scores = torch.cat([pyr.class_logits.max(dim=1)[0] for pyr in valid_pyramids])
        labels = torch.cat([pyr.class_logits.max(dim=1)[1] for pyr in valid_pyramids])
        box_masks = torch.cat([pyr.get_mask(3)[:, 1] for pyr in valid_pyramids], dim=0).unsqueeze(1)
        
        boxlist = BoxList(boxes, image_size[::-1], mode="xyxy")
        boxlist.add_field("class_logits", class_logits)
        boxlist.add_field("labels", labels)
        boxlist.add_field("scores", scores)
        boxlist.add_field("mask", box_masks)

        return boxlist


        cv2.imshow('pyr_list_to_boxlist show', show_img)
        cv2.waitKey()

        # (Pdb) result
        # [BoxList(num_boxes=50, image_width=1201, image_height=800, mode=xyxy), BoxList(num_boxes=50, image_width=1060, image_height=800, mode=xyxy)]
        # (Pdb) result[0]
        # BoxList(num_boxes=50, image_width=1201, image_height=800, mode=xyxy)
        # (Pdb) result[0].extra_fields.keys()
        # dict_keys(['scores', 'labels', 'mask', 'mask_scores'])
        # (Pdb) result[0].extra_fields['scores'].shape
        # torch.Size([50])
        # (Pdb) result[0].extra_fields['labels'].shape
        # torch.Size([50])
        # (Pdb) result[0].extra_fields['mask'].shape
        # torch.Size([50, 1, 28, 28])
        # (Pdb) result[0].extra_fields['mask_scores'].shape
        # torch.Size([50])


        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        这样复杂处理的主要原因是，要先NMS， 然后再比较NMS后留下的box互相竞争class 这时才分配labels
        eg 有个老二，在类A中被NMS比下去，它不一定死，还可以参加类B的NMS选修
        """

        # result = []
        # for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
        #     boxlist = BoxList(proposal, im_shape, mode="xyxy")
        #     boxlist.add_field("objectness", score)
        #     # 把proposal的框伸出图像的部分裁回图像
        #     boxlist = boxlist.clip_to_image(remove_empty=False)
        #     boxlist = remove_small_boxes(boxlist, self.min_size)
        #     # (((boxes[:,2]-boxes[:,0])>=0) & ((boxes[:,3]-boxes[:,1])>=0)).sum()
        #     boxlist = boxlist_nms(
        #         boxlist,
        #         self.nms_thresh,
        #         max_proposals=self.post_nms_top_n,
        #         score_field="objectness",
        #     )
        #     result.append(boxlist)



    def forward_singel_level(self, curr_level, inst_pyramids, x_curr, i, level_sizes, 
        target_support_pyramids, target_levels, losses_i):
        new_pos_limit = [100, 50, 50, 50, 50, 50, 50]
        new_pos_quota = 80
        if x_curr[[i]].abs().max() > 1e20 or torch.isnan(x_curr[[i]].max()):
        # if torch.isnan(x_curr[[i]].max()):
            print(curr_level, '\n', x_curr[[i]])
            import pdb; pdb.set_trace()
        # 生成 upsample mask，对现有的mask pyramids
        self.compute_mask(curr_level, x_curr[[i]], inst_pyramids)
        # TODO: 考虑其他的new_masks计算方法，比如说 multi target cross entropy loss 中的单一channel
        new_masks_minus = torch.cat([i_p.get_mask(curr_level)[:,[1]] - i_p.get_mask(curr_level)[:,[0]] for i_p in inst_pyramids], dim=1)
        new_masks_softmax = F.softmax(new_masks_minus,dim=1)
        # avg_sharing = 1.0 / len(inst_pyramids)
        # num_pixels = int(new_masks_softmax.shape[-1]*new_masks_softmax.shape[-2])
        # top_percent = new_masks_softmax.view(-1).topk(int(num_pixels*(1-0.3)))[0][-1].item()
        # max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(num_pixels-3)[0][-1].item()
        max_topk = new_masks_softmax.max(dim=1)[0].view(-1).topk(8, largest=False)[0][-1].item()
        # 这里非常的有趣，保证最少选拔8人，如果KOL话语权占不到5%，那就诞生新的KOL proposal
        # pending_thresh越高，新增的new_pos越多 所以 max_topk 应该是保底， 应该配合比例
        pending_thresh = max(0.02, max_topk)
        new_pos = torch.nonzero(new_masks_softmax[0].max(dim=0)[0] < pending_thresh)
        # if len(new_pos) > new_pos_limit[curr_level]:
        #     # import pdb; pdb.set_trace()
        #     raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(new_pos_limit[curr_level], largest=False)[1]
        #     new_pos_0 = raw_pos // x_curr.shape[-1]
        #     new_pos_1 = raw_pos % x_curr.shape[-1]
        #     new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
        # import pdb; pdb.set_trace()
        if len(inst_pyramids) + len(new_pos) > new_pos_quota:
            available_number = max(0, new_pos_quota - len(inst_pyramids))
            if available_number:
                raw_pos = new_masks_softmax.max(dim=1)[0].view(-1).topk(available_number, largest=False)[1]
                new_pos_0 = raw_pos // x_curr.shape[-1]
                new_pos_1 = raw_pos % x_curr.shape[-1]
                new_pos = torch.cat((new_pos_0.view(-1,1), new_pos_1.view(-1,1)), dim=1)
            else:
                new_pos = []

        new_occupy = 1.0*len(new_pos) / x_curr.shape[-2] / x_curr.shape[-1]
        # if new_occupy > 0.5 or len(new_pos) <8:
        # if new_occupy > 0.5 or len(new_pos) <8-1:
        #     print('new_occupy:{}| len(new_pos):{}'.format(new_occupy, len(new_pos)))
            # import pdb; pdb.set_trace()
        new_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in new_pos]
        self.compute_mask(curr_level, x_curr[[i]], new_pyramids, True)
        # 出清没有领地的pyramid 在所有pixel都进不了前3
        # 统计没有pyramid的targets
        # 额外惩罚霸占位置的pyramid，保护弱势应得的 pyramid
        merit_pyramids_idx = new_masks_softmax.topk(2, dim=1)[1].unique()
        # merit_pyramids_idx = new_masks_softmax.topk(3, dim=1)[1].unique()
        merit_pyramids = [inst_pyramids[i] for i in range(len(inst_pyramids)) if i in merit_pyramids_idx]

        if self.training:
            # 观测各级对于减少未匹配target的贡献
            target_len_before = sum([len(l) for l in target_support_pyramids])
            # target_len_1_before = sum([len(l) for l in target_support_pyramids_0])
            for reduce_i in range(len(inst_pyramids)):
                if reduce_i not in merit_pyramids_idx:
                    die_id = inst_pyramids[reduce_i].idx
                    die_target_idx = inst_pyramids[reduce_i].target_idx
                    if die_target_idx:
                        target_support_pyramids[die_target_idx].remove(die_id)
                        # target_support_pyramids_0[die_target_idx].remove(die_id)
            target_len_after = sum([len(l) for l in target_support_pyramids])
            # target_len_1_after = sum([len(l) for l in target_support_pyramids_0])
            # if target_len_1_before != target_len_1_after:

        inst_pyramids = merit_pyramids + new_pyramids
        self.log_dict.update({'pyr_num_l'+str(curr_level): len(inst_pyramids)})

        if self.training:
            self.match_target(curr_level, new_pyramids, target_levels, target_support_pyramids)
            loss = self.compute_loss(curr_level, inst_pyramids, target_levels, target_support_pyramids)
            losses_i.append(loss)
        
        return inst_pyramids

    def forward(self, image, targets=None):
        x_img = image.tensors
        # xs_r50 = self.r50(x_img)
        # import pdb; pdb.set_trace()
        xs_r50 = self.resnet50(x_img)

        xs_r50.append(self.res_layer_5(xs_r50[-1]))
        xs_r50.append(self.res_layer_6(xs_r50[-1]))

        # print('r50 max values:', [f.max().item() for f in xs_r50])

        N, _, img_size_h, img_size_w = x_img.shape
        device = x_img.device
        level_sizes = [tuple(f.shape[-2:]) for f in xs_r50[::-1]]

        losses = {}
        losses_0 = []
        losses_1 = []
        losses_2 = []
        losses_3 = []
        losses_4 = []
        losses_class = []
        test_masks = []
        target_support_pyramids = None
        for i in range(N):            
            InstancePyramid.inst_count = 0
            curr_level = 0
            x_curr = xs_r50[-1]
            init_pos = torch.nonzero(torch.ones_like(x_curr[0][0]))
            inst_pyramids = [InstancePyramid(pos, curr_level, level_sizes) for pos in init_pos]
            if x_curr[[i]].abs().max() > 1e19 or torch.isnan(x_curr[[i]].max()):
                import pdb; pdb.set_trace()
            self.compute_mask(curr_level, x_curr[[i]], inst_pyramids, True)
            self.log_dict.update({'pyr_num_l0': len(inst_pyramids)})
            target_levels = None
            if self.training:
                target_levels = self._init_target((img_size_h, img_size_w ), device, targets[i])
                target_support_pyramids = [[] for k in range(target_levels[7].shape[1])]
                # 统计 target 匹配
                self.match_target(0, inst_pyramids, target_levels, target_support_pyramids)
                
                loss_0 = self.compute_loss(0, inst_pyramids, target_levels, target_support_pyramids)
                losses_0.append(loss_0)
    
            # print(0, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)
            if xs_r50[-2][[i]].abs().max() > 1e20 or torch.isnan(xs_r50[-2][[i]].max()):
                print(1, '\n', xs_r50[-2][[i]])
                import pdb; pdb.set_trace()
            inst_pyramids = self.forward_singel_level(1, inst_pyramids, xs_r50[-2], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_1)
            # print(1, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)

            if xs_r50[-3][[i]].abs().max() > 1e20 or torch.isnan(xs_r50[-3][[i]].max()):
                print(2, '\n', xs_r50[-3][[i]])
                import pdb; pdb.set_trace()
            inst_pyramids = self.forward_singel_level(2, inst_pyramids, xs_r50[-3], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_2)
            # print(2, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)

            if xs_r50[-4][[i]].abs().max() > 1e20 or torch.isnan(xs_r50[-4][[i]].max()):
                print(3, '\n', xs_r50[-4][[i]])
                import pdb; pdb.set_trace()
            inst_pyramids = self.forward_singel_level(3, inst_pyramids, xs_r50[-4], i, level_sizes, 
                    target_support_pyramids, target_levels, losses_3)
            # print(3, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)

            class_loss = self.compute_class(inst_pyramids, target_levels)
            losses_class.append(class_loss)

            # inst_pyramids = self.forward_singel_level(4, inst_pyramids, xs_r50[-5], i, level_sizes, 
            #         target_support_pyramids, target_levels, losses_4)
            # print(4, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)

            # inst_pyramids = self.forward_singel_level(5, inst_pyramids, xs_r50[-6], i, level_sizes, 
            #         target_support_pyramids, target_levels, losses_5)
            # print(5, 'len(inst_pyramids)', len(inst_pyramids), target_support_pyramids)

            # import pdb; pdb.set_trace()


            if not self.training:
                # test_masks.append(inst_pyramids)
                # test_masks.append(self.pyr_list_to_boxlist(inst_pyramids))
                test_masks.append(self.pyr_list_to_boxlist_visual(inst_pyramids, image, image.image_sizes[i]))

            
        self.log_dict.update({'InstPyr_inst_count': InstancePyramid.inst_count})
        # import pdb; pdb.set_trace()
        losses['level_0']= sum(loss for loss in losses_0)
        losses['level_1']= sum(loss for loss in losses_1)
        losses['level_2']= sum(loss for loss in losses_2)
        losses['level_3']= sum(loss for loss in losses_3)
        losses['level_4']= sum(loss for loss in losses_4)
        losses['class']= sum(loss for loss in losses_class)

        return losses if self.training else test_masks

class InstancePyramid():
    inst_count = 0
    def __init__(self, pos, init_pub_level, level_sizes):
        self.idx = InstancePyramid.inst_count
        InstancePyramid.inst_count += 1
        self.init_level = init_pub_level
        self.level_sizes = level_sizes
        self.pos = pos
        self.masks = {}
        self.mask_logits = {}
        self.class_logits = None
        self.target_idx = None
        self.is_alive = True
        # torch.tensor(800.0/2**(7-self.init_level)).ceil().long().item()
        self.feature_scales = [7, 13, 25, 50, 100, 200]
        # self.gaussian_masks = self.generate_gaussian_masks()
        # import pdb; pdb.set_trace()
        # self.shared_gaussian_mask = self.shared_gaussian_masks()
        self.init_gaussian_mask = self.init_gaussian_masks()

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

    def generate_gaussian_masks_old(self):
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

    def get_feature_gaussian_mask_old(self, pub_level, feature_size):
        # gaussian_mask = self.gaussian_masks[pub_level - self.init_level]
        feature_size = tuple(feature_size)
        gaussian_mask = self.gaussian_masks[pub_level]
        level_pos = self.get_root_level_pos(pub_level)
        ctr = (self.feature_scales[pub_level]*2-1,)*2
        feature_g_mask = gaussian_mask[ctr[0]-level_pos[0]:ctr[0]-level_pos[0]+feature_size[0], \
            ctr[1]-level_pos[1]:ctr[1]-level_pos[1]+feature_size[1]]
        return feature_g_mask

    def shared_gaussian_masks(self):
        # feature_scales = [7, 13, 25, 50, 100, 200]
        xs = torch.arange(7*4)
        ys = torch.arange(7*4).view(-1,1)
        ln2 = torch.tensor(2.0, requires_grad=False).log()
        # shared_gaussian_mask = (-4*ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()
        shared_gaussian_mask = (-ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()
        return shared_gaussian_mask

    def init_gaussian_masks(self):
        xs = torch.arange(7*4)
        ys = torch.arange(7*4).view(-1,1)
        ln2 = torch.tensor(2.0, requires_grad=False).log()
        gaussian_mask_28 = (-ln2*((xs.float()-7*2+1)**2+(ys.float()-7*2+1)**2)/7**2).exp()

        init_gaussian_mask = torch.zeros(self.level_sizes[self.init_level])

        src_x0 = max(13-self.pos[0], 0)
        src_y0 = max(13-self.pos[1], 0)
        src_x1 = min(13+self.level_sizes[self.init_level][0]-self.pos[0], 28)
        src_y1 = min(13+self.level_sizes[self.init_level][1]-self.pos[1], 28)
        res_x0 = max(0, self.pos[0]-13)    #+1?
        res_y0 = max(0, self.pos[1]-13)
        res_x1 = res_x0+src_x1-src_x0
        res_y1 = res_y0+src_y1-src_y0

        init_gaussian_mask[res_x0:res_x1, res_y0:res_y1] = gaussian_mask_28\
            [src_x0:src_x1, src_y0:src_y1]

        return init_gaussian_mask

    def get_feature_gaussian_mask_v17(self, pub_level, feature_c0):
        level_pos = self.get_root_level_pos(pub_level)
        feature_g_mask = torch.zeros_like(feature_c0)

        src_x0 = max(13-level_pos[0], 0)
        src_y0 = max(13-level_pos[1], 0)
        # src_x1 = min(src_x0+feature_c0.shape[0], 28)
        # src_y1 = min(src_y0+feature_c0.shape[1], 28)
        src_x1 = min(13+feature_c0.shape[0]-level_pos[0], 28)
        src_y1 = min(13+feature_c0.shape[1]-level_pos[1], 28)
        res_x0 = max(0, level_pos[0]-13)    #+1?
        res_y0 = max(0, level_pos[1]-13)
        # res_x1 = min(feature_c0.shape[0], level_pos[0]+14+1)
        # res_y1 = min(feature_c0.shape[1], level_pos[1]+14+1)
        res_x1 = res_x0+src_x1-src_x0
        res_y1 = res_y0+src_y1-src_y0
        if feature_g_mask[res_x0:res_x1, res_y0:res_y1].shape != \
            self.shared_gaussian_mask[src_x0:src_x1, src_y0:src_y1].shape:
            print(feature_g_mask[res_x0:res_x1, res_y0:res_y1].shape)
            print(self.shared_gaussian_mask[src_x0:src_x1, src_y0:src_y1].shape)
            import pdb; pdb.set_trace()
        feature_g_mask[res_x0:res_x1, res_y0:res_y1] = self.shared_gaussian_mask[src_x0:src_x1, src_y0:src_y1]

        return feature_g_mask

    # def get_feature_gaussian_mask(self, pub_level, feature_c0):
    #     # import pdb; pdb.set_trace()
    #     feature_g_mask = F.interpolate(self.init_gaussian_mask[None,None,...]\
    #         .to(feature_c0.device), tuple(feature_c0.shape))[0,0]

    #     return feature_g_mask

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
    # checkpointer = DetectronCheckpointer(
    #     # cfg, model, optimizer, scheduler, output_dir, save_to_disk
    #     cfg, model.r50, optimizer, scheduler, output_dir, save_to_disk
    # )
    # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    # arguments.update(extra_checkpoint_data)

    # import pdb; pdb.set_trace()
    # # # # torch.save(model.r50.state_dict(), 'run/ImageNetPretrainedR50torch.pth')

    # model.r50.load_state_dict(torch.load('run/ImageNetPretrainedR50torch.pth'))
    # model_state = model.state_dict()
    # modify_dict = torch.load('run/try_v13/model_0090000.pth')['model']

    # modify_dict['chn256_4.weight'] = model_state['chn256_4.weight']
    # modify_dict['chn256_3.weight'] = model_state['chn256_3.weight']
    # modify_dict_keys = modify_dict.keys()
    # for k in model_state.keys():
    #     if k not in modify_dict_keys:
    #         modify_dict[k] = model_state[k]
    # model.load_state_dict(modify_dict)

    # model.load_state_dict(torch.load('run/try_v13/model_0090000.pth')['model'])

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
        None, # checkpointer,
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

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = transforms.functional.resize(image, size)
        return image


from torchvision import transforms
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list

class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
        display_text = False,
        display_scores = False,
    ):
        self.cfg = cfg.clone()
        self.model = MaskPyramids(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        self.model.load_state_dict(torch.load(cfg.MODEL.WEIGHT)['model'])

        self.transforms = self.build_transform()
        # self.transforms = build_transforms(cfg, is_train=False)

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        # self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        self.display_score = display_scores
        self.display_text = display_text

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = transforms.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = transforms.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = transforms.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                Resize(min_size, max_size),
                transforms.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # import pdb; pdb.set_trace()
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        if self.display_text:
            result = self.overlay_class_names(result, top_predictions, self.display_score)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            import pdb; pdb.set_trace()
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        if predictions.has_field("mask_scores"):
            scores = predictions.get_field("mask_scores")
        else:
            scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        # thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        # keep = torch.nonzero(scores > thresholds).squeeze(1)
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def compute_colors_for_labels_yolact(self, labels, class_color=False):
        """
        Simple function that adds fixed colors depending on the class
        """
        # colors = labels[:, None] * self.palette
        # colors = (colors % 255).numpy().astype("uint8")
        # colors = torch.cat([(class * 5) % len(COLORS) for class in labels])
        color_indice = [(labels[c] * 5) if class_color else c * 5 % len(COLORS) for c in range(len(labels))]
        # colors = torch.cat([COLORS[color_idx] for color_idx in color_indice])
        colors = [COLORS[color_idx] for color_idx in color_indice]
        return colors


    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist()
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)


        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances FCOS-MASK-SCORING-PLUS-R-50-FPN-MaskP3P5-new2xcontours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        #original
        # colors = self.compute_colors_for_labels(labels).tolist()

        #Detectron.pytorch for matplotlib colors
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)

        
        mask_img = np.copy(image)


        for mask, color in zip(masks, colors):
            # color_mask = color_list[color_id % len(color_list)]
            # color_id += 1

            # thresh = mask[0, :, :, None]
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_img = cv2.drawContours(mask_img, contours, -1, color, -1)


        # composite = image
        alpha = 0.45
        composite = cv2.addWeighted(image, 1.0 - alpha, mask_img, alpha, 0)
        # composite = cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0)

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, display_score=False):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        # labels = predictions.get_field("labels").tolist()
        labels = predictions.get_field("labels")
        # colors = self.compute_colors_for_labels(labels).tolist()
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)

        labels = labels.tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox


        # font_face = cv2.FONT_HERSHEY_COMPLEX
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        for box, score, label, color in zip(boxes, scores, labels, colors):
            x, y = box[:2]
            if display_score:
                template = "{}: {:.2f}"
                s = template.format(label, score)
            else:
                s = label
            text_w, text_h = cv2.getTextSize(s, font_face, font_scale, font_thickness)[0]
            text_pt = (x, y - 3)
            text_color = [255, 255, 255]
            # cv2.rectangle(image, (x,y), (x + text_w, y - text_h - 2), color,  -1)
            # cv2.putText(image, s, (x, y), font_face, font_scale, (255, 255, 255), 1)
            cv2.rectangle(image, (x,y), (x + text_w, y - text_h - 4), color,  -1) # mimicing yolact
            cv2.putText(image, s, text_pt, font_face, font_scale, text_color,font_thickness,  cv2.LINE_AA)

        return image

## mask_head/inference.py


def process_img(cfg):
    img_path_list = glob.glob("/home/xianr/TurboRuns/coco/val2014/*")
    img_path_list = img_path_list[:100]

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.07,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    )

    for img_path in img_path_list:
        start_time = time.time()
        img = cv2.imread(img_path)
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.imshow("COCO detections", composite)
        # cv2.imshow("COCO detections", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()





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
    cfg.merge_from_list(['OUTPUT_DIR', 'run/scratch_v21'])
    # cfg.merge_from_list(['MODEL.WEIGHT', ''])
    cfg.merge_from_list(['MODEL.WEIGHT', 'run/scratch_v21/model_0050000.pth'])
    cfg.merge_from_list(['SOLVER.IMS_PER_BATCH', 1])
    # cfg.merge_from_list(['SOLVER.BASE_LR', 1e-3])
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

    # model = train(cfg, args.local_rank, args.distributed)
    process_img(cfg)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()

