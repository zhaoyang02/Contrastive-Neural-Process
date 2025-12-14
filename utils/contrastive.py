import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Implementation of the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Args:
        z1: Batch of latent representations [Batch_Size, Dim] (View 1)
        z2: Batch of latent representations [Batch_Size, Dim] (View 2)
        temperature: Scaling factor
    
    Returns:
        loss: Scalar float
    """
    batch_size = z1.shape[0]
    
    # Normalize representations
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate to create a pool of 2*N samples
    # Shape: [2*B, Dim]
    out = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix (Cosine Similarity)
    # Shape: [2*B, 2*B]
    sim_matrix = torch.mm(out, out.t()) / temperature
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # Create labels for positive pairs
    # z1[i] matches z2[i] -> index i matches index i + batch_size
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size), # z1 matches corresponding z2
        torch.arange(0, batch_size)               # z2 matches corresponding z1
    ], dim=0).to(z1.device)
    
    # Compute Cross Entropy Loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss