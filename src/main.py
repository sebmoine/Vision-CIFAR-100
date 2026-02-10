import logging
import sys
import os
import pathlib

import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torchmetrics.classification import MulticlassAccuracy

import torch
import tqdm
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from src import data
from src import models
from src import scripts
from src.utils import losses, optim, run, log_checkpoint


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, val_loader, test_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    if config["logging"].get("wandb"):
        wandb.watch(model, log=None) #log="gradients, log_freq=100" en debug
        wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})


    # Build the loss
    logging.info("= Loss")
    loss = losses.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = log_checkpoint.generate_unique_logpath(logging_config["logdir"], logname)
    log_checkpoint.setup_logging(logdir, mode="train")

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {val_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = log_checkpoint.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    logging.info(f"Nombre de GPUs dispobnibles : {torch.cuda.device_count()}")
    logging.info(f"Utilisation de DataParallel: {'DataParallel' in str(type(model))}")

    run.fit(
        config,
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        device,
        model_checkpoint,
        logdir
    )



@torch.no_grad()
def test(config, weights_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    data_config = config["data"]
    model_config = config["model"]
    criterion = losses.get_loss(config["loss"])

    logdir = config["logging"]["logdir"] + '/' + str(weights_path).split('/')[-2]
    log_checkpoint.setup_logging(logdir, mode="test")

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")


    _, _, test_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)


    model = models.build_model(model_config, input_size, num_classes).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    top1  = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    top5  = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    macro = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)

    epoch_loss = 0
    num_samples = 0
    correct = 0
    confidence_sum = 0
    error_confidences = []

    for inputs, targets in test_loader:

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        preds = logits.argmax(1)
        matches = preds == targets
        correct += matches.sum().item()

        # Confiance moyennes des prédicitions (proba)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values
        confidence_sum += confidence.sum().item()

        # Confiance sur les erreurs
        preds = probs.argmax(dim=1)
        errors = preds != targets
        if errors.any():
            error_probs = probs[errors, preds[errors]]
            error_confidences.extend(error_probs.cpu().tolist())

        top1.update(preds, targets)
        top5.update(logits, targets)  # logits direct pour top-k
        macro.update(preds, targets)

        epoch_loss += loss.item() * inputs.size(0)
        num_samples += targets.size(0)


    epoch_loss /= num_samples
    accuracy = correct / num_samples
    avg_confidence = confidence_sum / num_samples
    if len(error_confidences) > 0: mean_conf_errors = sum(error_confidences) / len(error_confidences)
    else: mean_conf_errors = 0.0

    logging.info("====== TEST RESULTS ======")
    logging.info("Batch-Ponderated Loss\t: %.3f", epoch_loss)
    logging.info("Top-1 Accuracy\t\t\t: %.3f", top1.compute())
    logging.info("Top-5 Accuracy\t\t\t: %.3f", top5.compute())
    logging.info("Macro Accuracy\t\t\t: %.3f", macro.compute())
    logging.info("Binary Accuracy\t\t: %.3f", accuracy)
    logging.info("Average Confidence\t\t: %.3f", avg_confidence)
    logging.info("Average Errors' Confidence\t\t: %.3f", mean_conf_errors)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} <config.yaml|model_name> <train|test>")
        sys.exit(-1)

    arg = sys.argv[1]       #config.yaml|model_name
    command = sys.argv[2]   #train|test

    if command == "train":
        logging.info(f"Loading {arg}")
        config = yaml.safe_load(open(arg, "r"))
        train(config)

    elif command == "test":
        model_name = arg
        logging.info(f"Searching checkpoint for '{model_name}'...")
        model_dir = scripts.get_latest_model_dir(model_name)

        logging.info(f"Using checkpoint: {model_dir}")

        config_path, weights_path = scripts.get_checkpoint_files(model_dir)

        config = yaml.safe_load(open(config_path, "r"))

        test(config, weights_path)

    else:
        logging.error("Command must be 'train' or 'test'")
        sys.exit(-1)