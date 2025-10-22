import sys
sys.path.append('/Users/qianzhiyuan/GRPO/tiny_flow_matching')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.mlp import MLP
from dataset.synthetic_dataset import DatasetCheckerboard

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize training parameters
    batch_size = 512
    lr = 1e-3
    iterations = 30000
    hidden_dim = 128
    writer = SummaryWriter(log_dir=f"logs/cfm_mlp_checkerboard_bs{batch_size}_hd{hidden_dim}")

    # Initialize dataset and model
    dataset = DatasetCheckerboard(device=device)
    flow = MLP(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)

    for step in range(iterations):
        x_1 = dataset.sample(batch_size)  # Sample from target distribution
        x_0 = torch.randn_like(x_1).to(device)  # Sample from base distribution (standard normal)
        t = torch.rand(batch_size, 1).to(device)  # Sample time steps from [0, 1]

        # Compute the Conditional Flow Matching objective
        x_t = (1 - t) * x_0 + t * x_1  # \phi_t(x_0)
        dx_t = x_1 - x_0  # u_t(x|x_t)

        # Predict the vector field using the MLP
        pred_dx_t = flow(x_t, t)

        # Compute the loss
        loss = F.mse_loss(pred_dx_t, dx_t)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), step)

        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    writer.flush()
    writer.close()

    torch.save(flow.state_dict(), "checkpoints/flow_matching_mlp.pt")

if __name__ == "__main__":
    main()