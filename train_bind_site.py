"""Fine-tune ProRB for protein/RNA binding-site prediction.

The script follows the NAR experiment settings: two residue-level binary heads,
positive-class weight 15, labels 2/6 ignored, AdamW, and end-to-end fine-tuning
of the ProRB layers.  The upstream ESMC and RNA-FM encoders remain frozen as in
``pretrain.Model``.  There is no validation split; the best checkpoint is chosen
by training loss and the test split is evaluated once after training.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from binding_site_dataset import BindingSiteDataset
from pretrain import Model


class BindingSiteHead(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.fc(features).squeeze(-1)


class BindingSiteModel(nn.Module):
    def __init__(
        self,
        alpha=0.8,
        hidden_dim=64,
        pretrained_weights_path=None,
        rnafm_path=None,
        esmc_weights_path=None,
    ):
        super().__init__()
        self.model = Model(
            hidden_dim=hidden_dim,
            alpha=alpha,
            rnafm_path=rnafm_path,
            esmc_weights_path=esmc_weights_path,
        )
        self.rna_head = BindingSiteHead(hidden_dim)
        self.prot_head = BindingSiteHead(hidden_dim)
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)
        # These modules belong to the later autoregressive generation branch and
        # are neither present in the released checkpoint nor used in this task.
        self.model.pos_emb.requires_grad = False
        for module in (
            self.model.rna_norm_gen,
            self.model.token_emb,
            self.model.encoder_layer,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def load_pretrained_weights(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]

        expected = self.model.state_dict()
        compatible = {}
        ignored = []
        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("model."):
                key = key[len("model.") :]
            if key in expected and expected[key].shape == value.shape:
                compatible[key] = value
            else:
                ignored.append(key)

        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading ProRB: {unexpected}")
        generation_only_prefixes = (
            "pos_emb",
            "rna_norm_gen.",
            "token_emb.",
            "encoder_layer.",
        )
        required_missing = [
            key
            for key in missing
            if not key.startswith(generation_only_prefixes)
        ]
        if required_missing:
            preview = ", ".join(required_missing[:10])
            raise RuntimeError(
                f"Checkpoint does not contain the complete ProRB model; "
                f"{len(required_missing)} required keys are missing "
                f"(first keys: {preview})"
            )
        print(
            f"Loaded ProRB checkpoint {path} "
            f"({len(compatible)} model tensors; {len(ignored)} task tensors ignored; "
            f"{len(missing) - len(required_missing)} unused generation tensors absent)."
        )

    def forward(self, protein_input, rna_input, prot_pad, rna_pad):
        _, rna_embed, protein_embed, _, _ = self.model(
            rna_input,
            protein_input,
            prot_pad,
            rna_pad,
            mode="train",
        )
        return self.rna_head(rna_embed), self.prot_head(protein_embed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--pretrained_weights_path", type=Path, required=True)
    parser.add_argument("--rnafm_path", required=True)
    parser.add_argument("--esmc_weights_path", type=Path, required=True)
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--max_prot_len", type=int, default=1024)
    parser.add_argument("--max_rna_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=15.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; 0 runs all epochs.",
    )
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def valid_positions(logits, labels):
    mask = (labels != 2) & (labels != 6)
    return logits[mask].contiguous(), labels[mask].float().contiguous()


def binary_metrics(targets, probabilities, threshold):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities > threshold).astype(np.int64)
    has_both_classes = np.unique(targets).size == 2
    return {
        "acc": float(accuracy_score(targets, predictions)),
        "auroc": float(roc_auc_score(targets, probabilities))
        if has_both_classes
        else None,
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "auprc": float(average_precision_score(targets, probabilities))
        if has_both_classes
        else None,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }


def run_epoch(
    model,
    loader,
    device,
    criterion,
    threshold,
    optimizer=None,
    max_steps=0,
    global_step=0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batch_count = 0
    probabilities = {"rna": [], "protein": []}
    targets = {"rna": [], "protein": []}

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            prot_ids = batch["prot_input_ids"].to(device, non_blocking=True)
            rna_ids = batch["rna_input_ids"].to(device, non_blocking=True)
            prot_labels = batch["prot_bind"].to(device, non_blocking=True)
            rna_labels = batch["rna_bind"].to(device, non_blocking=True)
            prot_mask = batch["prot_padding_mask"].to(device, non_blocking=True)
            rna_mask = batch["rna_padding_mask"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            rna_logits, prot_logits = model(prot_ids, rna_ids, prot_mask, rna_mask)
            rna_logits, rna_targets = valid_positions(rna_logits, rna_labels)
            prot_logits, prot_targets = valid_positions(prot_logits, prot_labels)
            rna_loss = criterion(rna_logits, rna_targets)
            prot_loss = criterion(prot_logits, prot_targets)
            loss = rna_loss + prot_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step + 1}: {loss}")

            if training:
                loss.backward()
                optimizer.step()
                global_step += 1

            total_loss += loss.item()
            batch_count += 1
            probabilities["rna"].extend(torch.sigmoid(rna_logits).detach().cpu().tolist())
            probabilities["protein"].extend(
                torch.sigmoid(prot_logits).detach().cpu().tolist()
            )
            targets["rna"].extend(rna_targets.detach().cpu().tolist())
            targets["protein"].extend(prot_targets.detach().cpu().tolist())

            if training:
                print(
                    f"step={global_step} loss={loss.item():.6f} "
                    f"rna_loss={rna_loss.item():.6f} prot_loss={prot_loss.item():.6f}",
                    flush=True,
                )
            if training and max_steps and global_step >= max_steps:
                break

    if not batch_count:
        raise RuntimeError("The dataloader produced no batches.")
    result = {
        "loss": total_loss / batch_count,
        "steps": batch_count,
        "rna": binary_metrics(targets["rna"], probabilities["rna"], threshold),
        "protein": binary_metrics(
            targets["protein"], probabilities["protein"], threshold
        ),
    }
    return result, global_step


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    train_path = args.data_folder / args.train_csv
    test_path = args.data_folder / args.test_csv
    train_data = BindingSiteDataset(
        train_path,
        args.rnafm_path,
        max_prot_len=args.max_prot_len,
        max_rna_len=args.max_rna_len,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BindingSiteModel(
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        pretrained_weights_path=args.pretrained_weights_path,
        rnafm_path=args.rnafm_path,
        esmc_weights_path=args.esmc_weights_path,
    ).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"parameters: trainable={trainable:,} frozen={frozen:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    best_loss = math.inf
    best_path = args.output_folder / "best_model.pt"
    history_path = args.output_folder / "training_history.jsonl"
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        result, global_step = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            args.threshold,
            optimizer=optimizer,
            max_steps=args.max_steps,
            global_step=global_step,
        )
        record = {"epoch": epoch, "global_step": global_step, "train": result}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if result["loss"] < best_loss:
            best_loss = result["loss"]
            if not args.no_save:
                torch.save(model.state_dict(), best_path)
                print(f"Saved {best_path}")
        if args.max_steps and global_step >= args.max_steps:
            break

    if args.skip_test:
        return
    if not test_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    if not args.no_save:
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    test_data = BindingSiteDataset(
        test_path,
        args.rnafm_path,
        max_prot_len=args.max_prot_len,
        max_rna_len=args.max_rna_len,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_result, _ = run_epoch(
        model, test_loader, device, criterion, args.threshold
    )
    test_path_out = args.output_folder / "test_metrics.json"
    test_path_out.write_text(json.dumps(test_result, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_result}), flush=True)


if __name__ == "__main__":
    main()
