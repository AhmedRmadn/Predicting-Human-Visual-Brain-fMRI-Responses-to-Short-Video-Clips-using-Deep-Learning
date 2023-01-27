import torch
import numpy as np
from tqdm import tqdm
from runner.metric import vectorized_correlation


def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch, collect_output=True):
    losses = []
    targets_all = []
    outputs_all = []

    model.train()
    t = tqdm(loader)

    for i, sample in enumerate(t):
        optimizer.zero_grad()

        input = sample["vid_data"].to(args.device)
        target = sample["fmri"].to(args.device)
        weight = sample["weight"].to(args.device)

        output = model(input)
        loss = criterion(output, target, weight)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        
        target = target.cpu().numpy()
        output = output.detach().cpu().numpy()
        if collect_output:
            targets_all.extend(target)
            outputs_all.extend(output)
            output_score = vectorized_correlation(targets_all, outputs_all)
            output_loss = np.mean(losses)
        else:
            output_score = vectorized_correlation(target, output)
            output_loss = loss.item()

        t.set_description(
            f"Epoch {epoch}/{args.epochs} - Train loss: {output_loss:0.4f}, score: {output_score:0.4f}"
        )
  

    return targets_all, outputs_all, np.mean(losses), output_score


def validate(args, model, loader, criterion):
    losses = []
    targets_all = []
    outputs_all = []

    t = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(t):
            input = sample["vid_data"].to(args.device)
            target = sample["fmri"].to(args.device)

            output = model(input)
            loss = criterion(output, target)

            losses.append(loss.item())
            targets_all.extend(target.cpu().numpy())
            outputs_all.extend(output.detach().cpu().numpy())

            score = vectorized_correlation(targets_all, outputs_all)
            mean_loss = np.mean(losses)
            t.set_description(
                f"\t  - Valid loss: {mean_loss:0.4f}, score: {score:0.4f}"
            )

    return targets_all, outputs_all, np.mean(losses), score
