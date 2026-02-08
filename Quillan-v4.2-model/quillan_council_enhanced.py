import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ==========================================
# Data Structures for 12-Step Reasoning Trace
# ==========================================

@dataclass
class CouncilOutput:
    """
    Carries the standard tensor output for the forward pass,
    PLUS the 'Reasoning Trace' metadata for the Ethical Paradox Engine.
    """
    hidden_states: torch.Tensor      # The actual data for the next layer
    active_experts: torch.Tensor     # Indices of experts chosen (B, T, k)
    router_logits: torch.Tensor      # Raw router scores for audit (B, T, n_experts)
    consensus_score: torch.Tensor    # The final scalar judgment (0-1) from hierarchy

# ==========================================
# 1. The Single Expert (C1-C32)
# ========================================== 

class CouncilExpert(nn.Module):
    """
    Standard Feed-Forward Network acting as a single Council Member.
    Architecture: n_embd -> 4*n_embd -> n_embd (Standard Transformer FFN)
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. The Hierarchical Consensus Mechanism
# ==========================================

class HierarchicalConsensus(nn.Module):
    """
    The 'Judge' of the Council. 
    Reduces the aggregate expert output into a single scalar 'Truth/Ethical' score.
    Structure: n_embd -> 32 -> 16 -> 8 -> 1
    """
    def __init__(self, n_embd: int, council_layers: List[int] = [32, 16, 8, 1]):
        super().__init__()
        layers = []
        input_dim = n_embd
        
        # Build the reduction tree
        for hidden_dim in council_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
            
        # Final projection to scalar (no activation, raw logit)
        # We remove the last GELU to allow full dynamic range before sigmoid/gate
        self.net = nn.Sequential(*layers[:-1], nn.Linear(layers[-2].out_features, 1))

    def forward(self, x):
        # x: (B, T, n_embd) -> (B, T, 1)
        return self.net(x)

# ==========================================
# 3. The Main Sparse MoE Layer
# ==========================================

class CouncilMoELayer(nn.Module):
    """
    Quillan v4.2 Enhanced Council Layer.
    Integrates:
    1. Sparse Top-k Gating (Efficiency)
    2. Load Balancing (Expert Health)
    3. Hierarchical Consensus (Ethical Alignment)
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_council_experts  # 32
        self.top_k = 2                               # Standard Top-2
        self.n_embd = config.n_embd
        
        # A. The Router
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)
        
        # B. The Experts
        self.experts = nn.ModuleList([
            CouncilExpert(self.n_embd, config.dropout) 
            for _ in range(self.num_experts)
        ])
        
        # C. The Hierarchy (Consensus Gate)
        # Note: We apply this to the *result* of the experts to judge the output
        self.consensus = HierarchicalConsensus(self.n_embd, config.council_layers)
        
        # D. Consensus Influence Gate
        # Learnable parameter: How much does the Council's scalar judgment 
        # modulate the final output amplitude?
        self.consensus_weight = nn.Parameter(torch.ones(1)) 

    def forward(self, x: torch.Tensor) -> CouncilOutput:
        batch_size, seq_len, n_embd = x.shape
        flat_x = x.view(-1, n_embd)  # (B*T, n_embd)

        # 1. Routing
        router_logits = self.router(flat_x)  # (B*T, n_experts)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select Top-K
        # weights: (B*T, k), selected_indices: (B*T, k)
        weights, selected_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Normalize weights so they sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # 2. Expert Computation
        # We construct the output tensor
        final_output = torch.zeros_like(flat_x)
        
        # Loop over k selected experts (standard loop implementation for clarity)
        # (In highly optimized CUDA kernels, this is done differently)
        for k in range(self.top_k):
            # Get the index of the k-th selected expert for each token
            expert_indices = selected_indices[:, k]
            
            # For each unique expert index, compute and add
            for expert_idx in range(self.num_experts):
                # Find tokens that chose this expert as their k-th choice
                mask = (expert_indices == expert_idx)
                if mask.any():
                    # Extract tokens
                    token_inputs = flat_x[mask]
                    # Run Expert
                    expert_out = self.experts[expert_idx](token_inputs)
                    # Add weighted contribution
                    weight_factor = weights[mask, k].unsqueeze(-1)
                    final_output[mask] += expert_out * weight_factor

        # Reshape back to (B, T, n_embd)
        moe_output = final_output.view(batch_size, seq_len, n_embd)
        
        # 3. Hierarchical Consensus Check
        # The Council "reviews" the proposed output
        consensus_logit = self.consensus(moe_output)  # (B, T, 1)
        consensus_score = torch.sigmoid(consensus_logit) # 0 to 1 confidence
        
        # 4. Modulation
        # If consensus is low, we dampen the signal (or theoretically, reject it)
        # This enforces the "Internal Honesty" protocol
        modulated_output = moe_output * (1.0 + (consensus_score - 0.5) * self.consensus_weight)
        
        return CouncilOutput(
            hidden_states=modulated_output,
            active_experts=selected_indices.view(batch_size, seq_len, self.top_k),
            router_logits=router_logits.view(batch_size, seq_len, self.num_experts),
            consensus_score=consensus_score
        )