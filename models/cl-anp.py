import torch
from attrdict import AttrDict
from models.anp import ANP
from utils.contrastive import nt_xent_loss

class CLANP(ANP):
    def __init__(self, *args, contrastive_weight=0.1, temperature=0.5, **kwargs):
        """
        Contrastive Learning Attentive Neural Process (CLANP).
        Inherits from ANP and adds an auxiliary contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def split_context(self, x, y):
        """
        Randomly splits the available context points (x, y) into two disjoint sets.
        """
        batch_size, num_points, _ = x.shape
        
        # Create a random permutation of indices
        perm = torch.randperm(num_points)
        
        # Split indices into two halves
        mid = num_points // 2
        # Ensure at least 1 point in each split if possible
        if mid == 0 and num_points > 1:
             mid = 1
             
        idx1 = perm[:mid]
        idx2 = perm[mid:]
        
        # Gather data
        x1 = x[:, idx1, :]
        y1 = y[:, idx1, :]
        x2 = x[:, idx2, :]
        y2 = y[:, idx2, :]
        
        return x1, y1, x2, y2

    def forward(self, batch, num_samples=None, reduce_ll=True):
        # 1. Standard ANP Forward Pass (ELBO Calculation)
        outs = super().forward(batch, num_samples, reduce_ll)
        
        # Only compute contrastive loss during training
        if self.training:
            # 2. Contrastive Learning Step
            # Use 'batch.x' (all available points) to ensure maximum data for splitting
            x1, y1, x2, y2 = self.split_context(batch.x, batch.y)
            
            # Ensure valid splits
            if x1.shape[1] > 0 and x2.shape[1] > 0:
                # Pass through the Latent Encoder (lenc) specifically
                # Note: ANP has both 'denc' (deterministic) and 'lenc' (latent).
                # We specifically want to regularize the latent path 'z'.
                
                qz1 = self.lenc(x1, y1)
                z1 = qz1.loc 
                
                qz2 = self.lenc(x2, y2)
                z2 = qz2.loc 
                
                cl_loss = nt_xent_loss(z1, z2, temperature=self.temperature)
                
                outs.loss += self.contrastive_weight * cl_loss
                outs.cl_loss = cl_loss 
                
        return outs