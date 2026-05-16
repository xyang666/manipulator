"""
bc_pretrain.py
--------------
BC pretraining: train the SAC PhysicsInformedActor to match MPC expert actions.
Directly trains the actor with MSE loss on tanh(mean) output.
Saves a checkpoint compatible with --resume --reset_critic --reset_alpha.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.physics_policy import PhysicsInformedActor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/root/manipulator/data/mpc_scene9.npz")
    p.add_argument("--output", type=str, default="/root/manipulator/checkpoints/bc_pretrain_scene9.pt")
    p.add_argument("--state_dim", type=int, default=69)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--hidden_dims", type=str, default="256,256")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--task_scale", type=float, default=1.0)
    p.add_argument("--nullspace_scale", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # Load MPC data
    data = np.load(args.data)
    obs_data = torch.tensor(data["obs"], dtype=torch.float32)
    act_data = torch.tensor(data["act"], dtype=torch.float32)
    print(f"[BC] Loaded {len(obs_data)} pairs from {args.data}")
    print(f"[BC] Action range: [{act_data.min().item():.4f}, {act_data.max().item():.4f}]")

    # Normalize observations only (actions stay in original scale, within tanh range)
    obs_mean = obs_data.mean(dim=0)
    obs_std = obs_data.std(dim=0).clamp(min=1e-6)
    obs_norm = (obs_data - obs_mean) / obs_std

    dataset = TensorDataset(obs_norm, act_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train the actual PhysicsInformedActor (Tanh activations, squashed output)
    actor = PhysicsInformedActor(args.state_dim, args.action_dim, hidden_dims,
                                 task_scale=args.task_scale,
                                 nullspace_scale=args.nullspace_scale)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            mean, log_std = actor.forward(batch_obs)
            # actor.sample() returns tanh-squashed action
            # For BC: predict deterministically via tanh(mean)
            pred = torch.tanh(mean)
            loss = loss_fn(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 50 == 0:
            print(f"[BC] epoch {epoch:4d}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    print(f"[BC] Final loss: {best_loss:.6f}")

    # Set log_std to small negative value for low initial entropy
    with torch.no_grad():
        actor.log_std_head.weight.data.zero_()
        actor.log_std_head.bias.data.fill_(-0.5)

    ckpt = {
        "actor": actor.state_dict(),
        "obs_normalizer": {"mean": obs_mean.numpy(), "std": obs_std.numpy()},
        "metadata": {"bc_epochs": args.epochs, "bc_loss": best_loss,
                     "data": args.data, "state_dim": args.state_dim,
                     "action_dim": args.action_dim, "hidden_dims": str(hidden_dims)},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"[BC] Saved pretrained checkpoint to {args.output}")


if __name__ == "__main__":
    main()
