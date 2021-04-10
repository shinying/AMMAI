"""Usage: python train.py DATASET
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from config import get_config
from dataset import get_loader
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler
import backbones
import losses

torch.backends.cudnn.benchmark = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args, cfg):
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    
    log_root = logging.getLogger()
    init_logging(log_root, cfg.output)
    train_loader, val_loader = get_loader(cfg)
    
    print('Model save at', cfg.output)
    
    backbone = eval("backbones.{}".format(args.network))(
        False, dropout=cfg.dropout, fp16=cfg.fp16).to(args.device)

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(args.device)))
            logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")
            

    margin_softmax = eval("losses.{}".format(args.loss))()
    module_partial_fc = PartialFC(
        resume=args.resume, batch_size=cfg.batch_size, margin_softmax=margin_softmax, 
        num_classes=cfg.num_classes, sample_rate=cfg.sample_rate, 
        embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(train_loader) * cfg.num_epoch)
    logging.info("Total Step is: %d" % total_step)

    callback_logging = CallBackLogging(50, total_step, cfg.batch_size, None)
    callback_checkpoint = CallBackModelCheckpoint(cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        backbone.train()
        module_partial_fc.train()
        pred = []
        gt = []
        for step, (img, label) in enumerate(train_loader):
            img = img.to(args.device)
            global_step += 1
            features = F.normalize(backbone(img))
            x_grad, loss_v, logits = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16:
                features.backward(grad_scaler.scale(x_grad))
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler)
            pred.append(logits.cpu().argmax(dim=1))
            gt.append(label)
            
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_pfc.step()
        
        # Validation
        if (epoch+1) % 5 == 0: 
            backbone.eval()
            module_partial_fc.eval()
            loss_all = 0.
            val_acc = 0
            val_size = 0
            for (img, label) in val_loader:
                img = img.to(args.device)
                with torch.no_grad():
                    features = F.normalize(backbone(img))
                result, loss_v = module_partial_fc.evaluate(label, features)
                loss_all += loss_v.item() * len(img)
                val_acc += (result == label.cpu()).sum()
                val_size += len(img)
                
            train_acc = (torch.cat(pred) == torch.cat(gt)).float().mean()
            logging.info(f"Train acc: {train_acc:.2%}, " +
                         f"Val loss: {loss_all/val_size:.4f}, " +
                         f"Val acc: {val_acc/val_size:.2%}")


if __name__ == "__main__":
    set_seed()
    
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('dataset', default='APD2', help='dataset', 
                        choices=['APD', 'APD2', 'APD3'])
    parser.add_argument('--network', type=str, default='iresnet50', help='backbone network')
    parser.add_argument('--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = get_config(args.dataset)
    
    main(args, cfg)
