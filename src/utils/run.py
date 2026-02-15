import torch
import wandb
import logging
import tqdm
import matplotlib.pyplot as plt

from .log_checkpoint import EarlyStopper
from .transforms import Mixup

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    acc = 0
    epoch_loss = 0
    num_samples = 0
    mixup = Mixup(criterion, alpha=0.2)

    for inputs, targets in tqdm.tqdm(loader, leave=False):
        inputs  = inputs.to(device,non_blocking=True) #non_blocking : accelère les transferts CPU->GPU
        targets = targets.to(device,non_blocking=True)
        inputs = mixup.mixup_data(inputs, targets)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = mixup.mixup_criterion(logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = mixup.mixup_criterion(logits)
            loss.backward()
            optimizer.step()

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
    optimizer,
    device,
    checkpoint,
    logdir
    ):

    es_config = config["earlystopping"]
    if es_config["state"]:
        es_patience  = es_config["patience"]
        es_min_delta = es_config["min_delta"]
        early_stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)


    num_epochs   =  config["nepochs"]
    config_scheduler = config["scheduler"]

    train_losses = []
    val_losses   = []
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"]) #otherwise None

    if config_scheduler["ROPscheduler"]["state"]:
        lr_factor    = config_scheduler["ROPscheduler"]["factor"]
        lr_patience  = config_scheduler["ROPscheduler"]["patience"]
        lr_threshold = config_scheduler["ROPscheduler"]["threshold"]
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, threshold=lr_threshold)
    else :
        config_warmup = config_scheduler["Cosine"]["warmup"]
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                             start_factor=config_warmup["start_factor"],
                                                             end_factor=config_warmup["end_factor"],
                                                             total_iters=config_warmup["n_epochs"]
                                                             ) 
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      T_max=(num_epochs - 10),
                                                                      eta_min=1e-6
                                                                      )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                             schedulers=[warmup_scheduler, cosine_scheduler],
                                                             milestones=[config_warmup["n_epochs"]]
                                                             )

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        old_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch [{epoch+1}/{config['nepochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if config["logging"].get("wandb"):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": old_lr
            })

        if es_config["state"] & early_stopper.early_stop(val_loss):             
            break

        updated = checkpoint.update(val_loss)
        if updated:
            logging.info("New best model saved!")
            #wandb.save(str(checkpoint.savepath)) # If save in Cloud

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr: logging.info(f"LR reduced to {new_lr:.2e}")
        else:
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