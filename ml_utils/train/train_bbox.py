import os

import torch
from tqdm import tqdm

from ..utils.averager import Averager


def __send_to_device(images, targets, device):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def get_loss_val(model, images, targets, loss_hist):
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    loss_hist.send(loss_value)
    return loss_value, losses


def propagate(optimizer, losses):
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()


def train(model, tr_dl, val_dl, num_epochs, lr, momentum, weight_decay, step_size, gamma,
          output_dir=None, fn_best='weights_best.pth', fn_last='weights_last.pth'):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_hist = Averager()
    val_loss_hist = Averager()
    best_loss = 10 ** 10

    model.train()

    for epoch in range(num_epochs):
        loss_hist.reset()
        for images, targets, image_ids in tqdm(tr_dl):
            images, targets = __send_to_device(images, targets, device)
            loss_value, losses = get_loss_val(model, images, targets, loss_hist)
            propagate(optimizer, losses)

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss_hist.reset()
        with torch.no_grad():
            for images, targets, image_ids in val_dl:
                images, targets = __send_to_device(images, targets, device)
                get_loss_val(model, images, targets, val_loss_hist)

        print(f"Epoch #{epoch} tr loss: {loss_hist.value}, val loss: {val_loss_hist.value}")
        if output_dir is not None:
            torch.save(model.state_dict(), os.path.join(output_dir, fn_last))
        if val_loss_hist.value < best_loss:
            best_loss = val_loss_hist.value
            if output_dir is not None:
                torch.save(model.state_dict(), os.path.join(output_dir, fn_best))
    return model
