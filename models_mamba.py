import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import DropPath
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

PACKET_NUM = 5  # 5 packets constitutes an flow array

class StrideEmbed(nn.Module):
    def __init__(self, img_height=40, img_width=40, stride_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = img_height * img_width // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):#,drop_path=0.,
    #):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(  # 定义前向传播方法
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        """
        将输入传递给编码层的前向传播方法。

        参数:
            hidden_states: 输入序列 (必须提供)。
            residual: 输入的残差，如果 residual 为 None，hidden_states 直接作为残差使用。
        """
        # 如果不启用 fused_add_norm，则进行标准的 LayerNorm 操作
        if not self.fused_add_norm:
            # 计算残差: 如果传入了 residual，将其加到 hidden_states 上，否则将 hidden_states 作为 residual
            residual = (hidden_states + residual) if residual is not None else hidden_states

            # 将 residual 经过 norm 归一化
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

            # 如果 residual_in_fp32 为真，将 residual 转换为 float32 处理
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:  # 启用了 fused_add_norm 时，使用融合的归一化和加法操作
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # 使用 fused_add_norm_fn 处理 hidden_states 和 residual
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,  # 使用预归一化
                residual_in_fp32=self.residual_in_fp32,  # 在 fp32 中计算残差
                eps=self.norm.eps,  # 归一化时使用的 epsilon 值，防止除零错误
            )

        # 将归一化后的 hidden_states 传递给 mixer 模块进行进一步处理
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        # 返回处理后的 hidden_states 和 residual
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    # if_bimamba=False,
    # bimamba_type="none",
    # if_devide_out=False,
    # init_layer_scale=None,
):
    # if if_bimamba:
    #     bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    #mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    mixer_cls = partial(Mamba, layer_idx=layer_idx,**ssm_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block