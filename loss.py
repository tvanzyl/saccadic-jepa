import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from lightly.utils.dist import gather

class ReSALoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,                
    ):
        self.temp = temperature
        self.n_iterations = sinkhorn_iterations
        super().__init__()

    def cross_entropy(self, s, q):
        return - torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean()

    @torch.no_grad()
    def sinkhorn_knopp(self, scores, temp=0.05, n_iterations=3):
        Q = torch.exp(scores / temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,        
        emb: Tensor,
        emb_m: Tensor,
        f: Tensor,
        f_m: Tensor,

    ) -> Tensor:
        emb = F.normalize(emb)
        emb_m = F.normalize(emb_m)
        f = F.normalize(f)
        f_m = F.normalize(f_m)

        with torch.no_grad():
            assign = self.sinkhorn_knopp(f @ f_m.T, self.temp, self.n_iterations)        
        emb_sim = emb @ emb_m.T / self.temp
        loss = self.cross_entropy(emb_sim, assign)
        return loss
    
class JSLoss(torch.nn.Module):

    def __init__(
        self,
        lambda_param: float = 0.1,
        gather_distributed: bool = False,
        eps: float = 0.0001,
    ):
        """Initializes the VICRegLoss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(JSLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param                
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # Invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # Gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        # Total VICReg loss
        loss = inv_loss + self.lambda_param*cov_loss

        return loss

def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).

    Returns:
        The computed VICReg invariance loss.
    """
    return -F.cosine_similarity(F.normalize(x), F.normalize(y)).mean()


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x: Tensor with shape (batch_size, ..., dim).

    Returns:
          The computed VICReg covariance loss.
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)

    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x).pow(2) / (batch_size - 1)    
    # var = torch.diag(cov_matrix)
    # inv_std = 1.0/torch.sqrt(var)
    # inv_std_matrix = torch.diag(inv_std)
    # cor_matrix = torch.matmul(torch.matmul(inv_std_matrix, cov_matrix), inv_std_matrix)

    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()