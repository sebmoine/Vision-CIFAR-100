import torch
import wandb
import logging
import tqdm
import matplotlib.pyplot as plt

from .log_checkpoint import EarlyStopper
from .transforms import Mixup
from .SAM import *


def train_one_epoch_mixup(model, loader, criterion, optimizer, device, scaler):
    model.train()

    acc = 0
    epoch_loss = 0
    num_samples = 0

    mix_data    = criterion.mixup_data
    criterion   = criterion.mixup_criterion

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs  = inputs.to(device,non_blocking=True) #non_blocking : accelère les transferts CPU->GPU
        targets = targets.to(device,non_blocking=True)
        inputs = mix_data(inputs, targets)

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(1)
        acc += (preds == targets).sum().item()
        num_samples += targets.size(0)

    epoch_loss = epoch_loss / num_samples
    epoch_acc = acc / num_samples

    return epoch_loss, epoch_acc

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    acc = 0
    epoch_loss = 0
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs  = inputs.to(device,non_blocking=True) #non_blocking : accelère les transferts CPU->GPU
        targets = targets.to(device,non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with torch.amp.autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(1)
        acc += (preds == targets).sum().item()
        num_samples += targets.size(0)

    epoch_loss = epoch_loss / num_samples
    epoch_acc = acc / num_samples

    return epoch_loss, epoch_acc




@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    epoch_loss = 0.0
    acc = 0
    num_samples = 0

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs = inputs.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        epoch_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(1)
        acc += (preds == targets).sum().item()
        num_samples += targets.size(0)

    epoch_loss = epoch_loss / num_samples
    epoch_acc = acc / num_samples

    return epoch_loss, epoch_acc

def fit(
    config,
    model,
    train_loader,
    val_loader,
    criterion,
    base_optimizer,
    device,
    checkpoint,
    logdir
    ):

    if config["optim"]["SAM"]:
        base_optimizer = SAM(model.parameters(), base_optimizer)

    num_epochs   =  config["nepochs"]
    config_scheduler = config["scheduler"]

    train_losses = []
    val_losses   = []
    scaler = torch.amp.GradScaler('cuda', enabled=config["amp"]) #otherwise None

    if config_scheduler["ROPscheduler"]["state"]:
        lr_factor    = config_scheduler["ROPscheduler"]["factor"]
        lr_patience  = config_scheduler["ROPscheduler"]["patience"]
        lr_threshold = config_scheduler["ROPscheduler"]["threshold"]
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, mode='min', factor=lr_factor, patience=lr_patience, threshold=lr_threshold)
    else :
        config_warmup = config_scheduler["Cosine"]["warmup"]
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(base_optimizer,
                                                             start_factor=config_warmup["start_factor"],
                                                             end_factor=config_warmup["end_factor"],
                                                             total_iters=config_warmup["n_epochs"]
                                                             ) 
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer,
                                                                      T_max=(num_epochs - 10),
                                                                      eta_min=1e-6
                                                                      )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(base_optimizer,
                                                             schedulers=[warmup_scheduler, cosine_scheduler],
                                                             milestones=[config_warmup["n_epochs"]]
                                                             )
    if config["mixup"]:
        mixup_criterion = Mixup(criterion, alpha=0.2)
    else :
        mixup_criterion = criterion

    for epoch in range(num_epochs):

        if epoch < 50 or epoch >= 250:  #Mixup OFF
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, base_optimizer, device, scaler)
        else:                           #Mixup ON
            train_loss, train_acc = train_one_epoch_mixup(model, train_loader, mixup_criterion, base_optimizer, device, scaler)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        old_lr = base_optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch [{epoch+1}/{config['nepochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": old_lr
        })

        updated = checkpoint.update(val_loss)
        if updated:
            logging.info("New best model saved!")
            #wandb.save(str(checkpoint.savepath)) # If save in Cloud

        lr_scheduler.step()

    plt.figure()
    plt.plot(train_losses, c="red", label="train loss")
    plt.plot(val_losses, c="blue", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{logdir}/training_losses.png')
    plt.close()