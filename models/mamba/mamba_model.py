from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args) # Mamba块
        self.norm = RMSNorm(args.d_model) # RMS归一化

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (batch size, sequence length, hidden dim)
        Returns:
            output: shape (b, l, d)  (batch size, sequence length, hidden dim)
        """
        output = self.mixer(self.norm(x)) + x # [Norm -> Mamba -> Add]
        return output

class RMSNorm(nn.Module):
    '''
    均方根归一化： RMS normalization
    '''
    def __init__(self,
                 d_model: int, # hidden dim
                 eps: float = 1e-5): # 防止除以零的小数值
        super().__init__()
        self.eps = eps
        # weight: 可学习的参数，调整归一化后的值
        self.weight = nn.Parameter(torch.ones(d_model)) # 初始值为大小为d_model的张量，每个元素的值都是1

    def forward(self, x):
        '''
        :param x: 输入张量
        :return: output 均方根规划化后的值

        RMS的计算步骤：
        Step1: 计算每个样本的均方根值
            Step1.1: 先计算x的平方
            Step1.2: 沿着最后一个维度（通常是特征维度）计算平均值，并加上一个很小的数eps
            Step1.3: 最后取平方根
        Step2: 对每个样本进行归一化
            Step2.1：每个特征值除以其所在样本的均方根值
            Step2.2: 最后乘以可以学习的权重weight,得到最终的输出
        '''
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size,
                                      args.d_model)  # 词嵌入，其中包含 `args.vocab_size` 个不同的词或标记，每个词嵌入的维度为 `args.d_model`
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])  # n_layer个ResidualBlock
        self.norm_f = RMSNorm(args.d_model)  # RMS归一化

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
        # See "Weight Tying" paper

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)  # (b,l,d)  生成词嵌入

        for layer in self.layers:  # 通过n_layer个ResidualBlock
            x = layer(x)  # (b,l,d)

        x = self.norm_f(x)  # (b,l,d)
        logits = self.lm_head(x)  # (b,l,vocab_size)

        return logits
