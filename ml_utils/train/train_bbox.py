import argparse
import json
import os
import time

import torch
import wandb
from tqdm import tqdm

from ..utils.summary_stats import Averager, accuracy, summarize_accuracy
from ..utils.utils import remove_overlapping_boxes_torch, get_boxes_above_threshold


def __send_to_device(images, targets, device):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def get_loss_val(model, images, targets, loss_hist):
    model.train()
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_hist.send(loss_value)
    return loss_value, losses


def propagate(optimizer, losses):
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()


def add_accuracy(model, images, targets, accuracy_df, config):
    model.eval()
    outputs = model(images)

    for i in range(len(outputs)):
        bboxes, scores = get_boxes_above_threshold(outputs[i], config.detection_thr)
        bboxes = remove_overlapping_boxes_torch(bboxes, scores, config.overlap_thr).data.cpu().numpy()
        gt_boxes = targets[i]['boxes'].data.cpu().numpy()
        accuracy_df.append(accuracy(bboxes, gt_boxes, image_id='', dist_thr=config.dist_thr))
    return accuracy_df


def train(model, tr_dl, val_dl, config, log_progress=False, model_dir=None, model_name=None,
          fn_best='weights_best.pth', fn_last='weights_last.pth', fn_conf='config.json'):
    if model_dir is not None:
        if log_progress:
            model_name = wandb.run.name
        elif model_name is None:
            model_name = str(time.time())
        os.makedirs(os.path.join(model_dir, model_name), exist_ok=True)
        with open(os.path.join(model_dir, model_name, fn_conf), 'w') as f:
            json.dump(config, f)
    config = argparse.Namespace(**config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    loss_hist = Averager()
    val_loss_hist = Averager()
    best_loss = 10 ** 10

    for epoch in range(config.num_epochs):
        loss_hist.reset()
        model.train()
        for images, targets, image_ids in tqdm(tr_dl):
            images, targets = __send_to_device(images, targets, device)
            loss_value, losses = get_loss_val(model, images, targets, loss_hist)
            propagate(optimizer, losses)
            wandb.log({'loss': loss_value})

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss_hist.reset()
        accuracy_df = []
        with torch.no_grad():
            for images, targets, image_ids in val_dl:
                images, targets = __send_to_device(images, targets, device)
                get_loss_val(model, images, targets, val_loss_hist)
                accuracy_df = add_accuracy(model, images, targets, accuracy_df, config)

        wandb.log(summarize_accuracy(accuracy_df))
        wandb.log({'training loss': loss_hist.value,
                   'validation loss': val_loss_hist.value,
                   'epoch': epoch})

        print(f"Epoch #{epoch} tr loss: {loss_hist.value}, val loss: {val_loss_hist.value}")
        if model_dir is not None:
            torch.save(model.state_dict(), os.path.join(model_dir, model_name, fn_last))
        if val_loss_hist.value < best_loss:
            best_loss = val_loss_hist.value
            if model_dir is not None:
                torch.save(model.state_dict(), os.path.join(model_dir, model_name, fn_best))

    return model
