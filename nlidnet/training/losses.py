import torch
from torch.nn import functional as F


def calculate_variance_loss(
    attention_probs: torch.Tensor, T: int, k: int, device: torch.device
) -> torch.Tensor:
    B, L = attention_probs.shape
    attn_local = attention_probs.view(B, T, k)
    i_grid = torch.arange(1, k + 1, device=device, dtype=torch.float32).view(1, 1, k)
    mu = torch.sum(i_grid * attn_local, dim=-1, keepdim=True)
    l_var = torch.sum(((i_grid - mu) ** 2) * attn_local, dim=-1)
    return torch.mean(l_var)


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    device = features.device
    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)
    logits = torch.matmul(features, features.T) / temperature

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    batch_size = features.shape[0]
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    positives_mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_sample = positives_mask.sum(dim=1)
    valid = positives_per_sample > 0

    if valid.sum() == 0:
        return torch.zeros((), device=device, dtype=features.dtype)

    mean_log_prob_pos = (positives_mask * log_prob).sum(dim=1) / (
        positives_per_sample + 1e-12
    )
    return -mean_log_prob_pos[valid].mean()
