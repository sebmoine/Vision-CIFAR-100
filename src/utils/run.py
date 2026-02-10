import torch
import wandb
import logging
import tqdm
import matplotlib.pyplot as plt

from .log_checkpoint import EarlyStopper

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    acc = 0
    epoch_loss = 0
    num_samples = 0
    for inputs, targets in tqdm.tqdm(loader, leave=False):

        inputs  = inputs.to(device,non_blocking=True) #Accelère les transferts CPU->GPU
        targets = targets.to(device,non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(inputs)
            loss = criterion(logits, targets)

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

    es_patience  = config["earlystopping"]["patience"]
    es_min_delta = config["earlystopping"]["min_delta"]
    lr_factor    = config["ROPscheduler"]["factor"]
    lr_patience  = config["ROPscheduler"]["patience"]
    lr_threshold  = config["ROPscheduler"]["threshold"]

    train_losses = []
    val_losses   = []
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"]) #otherwise None
    early_stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, threshold=lr_threshold)
    best_val_loss = float("inf")

    for epoch in range(config["nepochs"]):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
                "val_acc": val_acc
            })

        if early_stopper.early_stop(val_loss):             
            break

        updated = checkpoint.update(val_loss)
        if updated:
            logging.info("New best model saved!")
            #wandb.save(str(checkpoint.savepath)) #Pour sauvegarder dans le Cloud
        
        old_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr: logging.info(f"LR reduced to {new_lr:.2e}")

    plt.figure()
    plt.plot(train_losses, c="red", label="train loss")
    plt.plot(val_losses, c="blue", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{logdir}/training_losses.png')
    plt.close()