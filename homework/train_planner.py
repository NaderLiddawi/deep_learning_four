"""
Usage:
    python3 -m homework.train_planner --model mlp_planner
    python3 -m homework.train_planner --model transformer_planner
    python3 -m homework.train_planner --model cnn_planner
"""

import argparse

import torch

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import CNNPlanner, MLPPlanner, TransformerPlanner, save_model


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred:   (B, n, 2)
        target: (B, n, 2)
        mask:   (B, n)  bool — True for valid waypoints
    """
    loss = (pred - target).abs()          # (B, n, 2)
    loss = loss * mask[..., None]         # zero out invalid
    return loss.sum() / mask.sum().clamp(min=1)


def run_epoch(model, loader, optimizer, device, model_name):
    """One pass over loader. If optimizer is None, eval mode."""
    is_train = optimizer is not None
    model.train(is_train)
    metric = PlannerMetric()
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            # move tensors to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # forward
            if model_name in ("mlp_planner", "transformer_planner"):
                pred = model(track_left=batch["track_left"], track_right=batch["track_right"])
            else:
                pred = model(image=batch["image"])

            loss = masked_l1_loss(pred, batch["waypoints"], batch["waypoints_mask"])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metric.add(pred.detach(), batch["waypoints"], batch["waypoints_mask"])
            total_loss += loss.item()

    results = metric.compute()
    results["loss"] = total_loss / len(loader)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str,   required=True,
                        choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--data_path",  type=str,   default="drive_data")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # ── model ────────────────────────────────────────────────────────────────
    model_cls = {"mlp_planner": MLPPlanner,
                 "transformer_planner": TransformerPlanner,
                 "cnn_planner": CNNPlanner}[args.model]
    model = model_cls().to(device)

    # ── data pipeline ────────────────────────────────────────────────────────
    # MLP + Transformer only need track boundaries (no image) → faster loading
    pipeline = "state_only" if args.model in ("mlp_planner", "transformer_planner") else "default"

    train_loader = load_data(
        f"{args.data_path}/train",
        transform_pipeline=pipeline,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = load_data(
        f"{args.data_path}/val",
        transform_pipeline=pipeline,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ── optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # cosine LR decay from lr → lr/10 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10
    )

    # ── training loop ────────────────────────────────────────────────────────
    best_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_res = run_epoch(model, train_loader, optimizer, device, args.model)
        val_res   = run_epoch(model, val_loader,   None,      device, args.model)
        scheduler.step()

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train_loss={train_res['loss']:.4f} | "
            f"val long={val_res['longitudinal_error']:.4f}  "
            f"lat={val_res['lateral_error']:.4f}  "
            f"l1={val_res['l1_error']:.4f}"
        )

        # save best
        if val_res["l1_error"] < best_l1:
            best_l1 = val_res["l1_error"]
            save_model(model)
            print(f"  ✓ saved (best l1={best_l1:.4f})")

    print(f"\nDone. Best val l1_error = {best_l1:.4f}")


if __name__ == "__main__":
    main()
