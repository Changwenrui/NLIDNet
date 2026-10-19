import torch
from torch import nn
from torch.nn import functional as F
from nlidnet.models.temporal.positional_encoding import (
    tAPE,
    AbsolutePositionalEncoding,
    LearnablePositionalEncoding,
)
from nlidnet.models.temporal.attention import (
    Attention,
    Attention_Rel_Scl,
    Attention_Rel_Vec,
)


class TemporalEncoder(nn.Module):

    def __init__(self, config, k: int):
        super().__init__()
        channel_size, seq_len = (config["Data_shape"][1], config["Data_shape"][2])
        emb_size = config["emb_size"]
        num_heads = config["num_heads"]
        dim_ff = config["dim_ff"]
        self.Fix_pos_encode = config["Fix_pos_encode"]
        self.Rel_pos_encode = config["Rel_pos_encode"]
        self.k = k
        self.seq_len = seq_len
        assert (
            self.seq_len % self.k == 0
        ), f"seq_len={self.seq_len} 必须能被 k={self.k} 整除，才能按段做 softmax"
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding="same"),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(
                emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding="valid"
            ),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )
        if self.Fix_pos_encode == "tAPE":
            self.Fix_Position = tAPE(
                emb_size, dropout=config["dropout"], max_len=seq_len
            )
        elif self.Fix_pos_encode == "Sin":
            self.Fix_Position = AbsolutePositionalEncoding(
                emb_size, dropout=config["dropout"], max_len=seq_len
            )
        elif config["Fix_pos_encode"] == "Learn":
            self.Fix_Position = LearnablePositionalEncoding(
                emb_size, dropout=config["dropout"], max_len=seq_len
            )
        if self.Rel_pos_encode == "eRPE":
            self.attention_layer = Attention_Rel_Scl(
                emb_size, num_heads, seq_len, config["dropout"]
            )
        elif self.Rel_pos_encode == "Vector":
            self.attention_layer = Attention_Rel_Vec(
                emb_size, num_heads, seq_len, config["dropout"]
            )
        else:
            self.attention_layer = Attention(emb_size, num_heads, config["dropout"])
        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-05)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-05)
        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config["dropout"]),
        )
        self.att_proj = nn.Linear(emb_size, 1)

    def forward(self, x):
        B, C, L = x.shape
        assert (
            L == self.seq_len
        ), f"输入长度 L={L} 与配置中的 seq_len={self.seq_len} 不一致"
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != "None":
            x_src_pos = self.Fix_Position(x_src)
            att_tok = x_src + self.attention_layer(x_src_pos)
        else:
            att_tok = x_src + self.attention_layer(x_src)
        att_tok = self.LayerNorm(att_tok)
        out = att_tok + self.FeedForward(att_tok)
        out = self.LayerNorm2(out)
        logits = self.att_proj(out).squeeze(-1)
        G = self.seq_len // self.k
        logits_group = logits.view(B, G, self.k)
        att_group = F.softmax(logits_group, dim=-1)
        att = att_group.view(B, self.seq_len)
        return logits
