import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int = 2,
            time_dim: int = 1,
            hidden_dim: int = 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
            self,
            x_t: torch.Tensor, # [B, 2]
            t: torch.Tensor # [B, 1]
    ) -> torch.Tensor:
        """
        Time-step conditional flow-matching MLP.

        Args:
            x_t: Noisy data at time t, shape [B, 2].
            t: Time steps, shape [B, 1].    
        Returns:
            output: Predicted vector field, shape [B, 2]. You can 
                input any sample and return the corresponding vector
                for vector field is actually continuous.
        """
        x_t = x_t.reshape(-1, self.input_dim) # [B, 2]
        t = t.reshape(-1, self.time_dim).float() # [B, 1]
        h = torch.cat([x_t, t], dim=-1) # [B, 3]
        output = self.layers(h) # [B, 2]
        
        return output


        