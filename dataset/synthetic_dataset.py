import torch

class DatasetCheckerboard():
    """Generate a synthetic dataset of 2D points distributed in a checkboard pattern."""
    def __init__(
            self,
            dim: int = 2,
            device: torch.device = torch.device("cpu"),
    ):
        self.dim = dim
        self.device = device

    def sample(
            self,
            n: int
    ) -> torch.Tensor:
        """
        Generate `n` numbers of 2D points.
        
        Args:
            n: Number of points[samples] in dataset. 

        Returns:
            data: Tensor of shape (n, 2), distribute like checkboard. 
        """
        x1 = torch.rand(n) * 4 - 2
        x2 = torch.rand(n) - torch.randint(high=2, size=(n,)) * 2
        x2 = x2 + (torch.floor(x1) % 2)
        data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

        return data.float().to(self.device)