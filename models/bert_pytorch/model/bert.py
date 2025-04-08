import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.masked import masked_tensor
from torch.onnx.symbolic_opset9 import tensor

from .transformer import TransformerBlock

def randow_mask_bert(mask_ratio, x):
    N,L,Emb = x.size()
    len_mask = int (L * mask_ratio)
    mask_embeding = torch.zeros(1,1,Emb,device=x.device)
    mask = torch.zeros((N, L), dtype=torch.bool, device=x.device)

    #len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)  # 噪声用于随机打乱

    # 排序噪声以获得要mask的 patches 的索引
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 找到 len_mask 个 patches
    ids_mask = ids_shuffle[:, :len_mask]
    # 使用 scatter_ 方法将 ids_mask 中的位置设置为 True，表示这些位置将被掩码
    for i in range(N):
        mask[i, ids_mask[i]] = True
    # print(mask[0])
    # print(ids_mask[0])
    masked_x = x.clone()
    masked_x[mask] = mask_embeding
    return masked_x, mask

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, args,hidden=768, n_layers=12, attn_heads=12):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.cls_token._no_weight_decay = True
        self.pos_encoding = PositionalEncoding(hidden,args.pack_len+1)
        self.mask_tensor = nn.Parameter(torch.zeros(1, 1, hidden)*0.01)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        #self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        # self.transformer_blocks = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         hidden,
        #         attn_heads,
        #         dim_feedforward=hidden*4,
        #         dropout=dropout,
        #         activation=F.gelu,
        #         batch_first=True,
        #     ),
        #     n_layers,
        #     norm=nn.LayerNorm(hidden),
        #     mask_check=False,
        # )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, args.bert_dropout) for _ in range(n_layers)])

    def forward(self, x, mask_ratio):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size, seq_len = x.shape[0], x.shape[1]+1
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        if mask_ratio != 0:
            x, mask = randow_mask_bert(mask_ratio, x)
            mask_tensor = self.mask_tensor.expand_as(x)
            x[mask] = mask_tensor[mask]
        else:
            mask = None

        x = torch.cat((cls_tokens, x), dim=1)
        # running over multiple transformer blocks
        # for transformer in self.transformer_blocks:
        #     x = transformer.forward(x, mask=None)
        # 获取并扩展位置编码
        # pos_encoding = self.pos_encoding[:, :seq_len, :]  # 截取到实际序列长度
        # pos_encoding = pos_encoding.expand(batch_size, -1, -1)

        # 将位置编码添加到输入
        x = x + self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        if mask_ratio != 0:
            return x, mask
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden, max_len=17):
        super(PositionalEncoding, self).__init__()

        # 创建一个足够大的位置编码矩阵
        pe = torch.zeros(max_len, hidden)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加批次维度，并将其注册为非训练参数
        pe = pe.unsqueeze(0)  # (1, max_len, hidden)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden]
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]  # 截取到实际的序列长度

