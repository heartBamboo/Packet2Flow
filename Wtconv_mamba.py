
from torch import nn
import torch
from models.WTConv.wtconvnext.wtconvnext import wtconvnext_tiny
from models_net_mamba import NetMamba
from utils import ClassificationHead

class wtconv_mamba(nn.Module):
    """
    混合模型
    """
    def __init__(self, args,mode,hidden=256, n_layers=12, attn_heads=8):

        super().__init__()
        self.mode = mode
        self.wtconv = wtconvnext_tiny(mode,args)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        # self.mamba = Mamba(d_model=hidden, d_state=128, d_conv=4, expand=2,
        #                        dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
        #                        dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
        #                        layer_idx=None, device=None, dtype=None)
        if mode == 'to_pretrain':
            self.mamba = NetMamba(is_pretrain=True, img_size=16, stride_size=4, embed_dim=256, depth=4,decoder_embed_dim=128, decoder_depth=2)
        else:
            self.mamba = NetMamba(is_pretrain=False, img_size=16, stride_size=4, embed_dim=256, depth=4,decoder_embed_dim=128, decoder_depth=2)
        #self.mamba = NetMamba(is_pretrain=True, img_size=16, stride_size=4, embed_dim=256, depth=4,decoder_embed_dim=128, decoder_depth=2)
        self.pack_len=args.pack_len
        self.confidence=args.confidence
        if mode != 'to_pretrain':
            self.head = ClassificationHead(input_dim=256, num_classes=args.num_classes, dropout_rate=args.head_dropout)
            if args.confidence:
                self.confidence_generator = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                    nn.Sigmoid()
                )

    def forward_feature(self, input, wt_mask_ratio,mask_ratio):
        wt_conv_output = self.wtconv(input, wt_mask_ratio)
        mamba_input = self.pool(wt_conv_output).view(-1, self.pack_len, wt_conv_output.shape[1])
        if self.mode == 'to_pretrain':
            #mamba_input1,mask=randow_mask_mamba(mask_ratio, mamba_input)
            #print(mamba_input1)
            mamba_output,mask = self.mamba(mamba_input)
            #print(mamba_output)
            return wt_conv_output, mamba_input, mamba_output,mask
        elif self.mode == 'to_finetune':
            mamba_output = self.mamba(mamba_input, mask_ratio)
            return mamba_output
        elif self.mode == 'just_evaluate':
            mamba_output = self.mamba(mamba_input, 0)
            return mamba_output

    def forward_feature_only_wtconv(self, input, wt_mask_ratio,bert_mask_ratio):
        wt_conv_output = self.wtconv(input, wt_mask_ratio)
        wt_conv_output = self.pool(wt_conv_output).view(-1, self.pack_len,wt_conv_output.shape[1]).mean(dim=1)
        return wt_conv_output


    def forward_head(self, input):
        output = self.head(input)
        return output


    def forward(self, input, wt_mask_ratio,bert_mask_ratio):
        if self.mode == 'to_pretrain':
            wt_conv_output, bert_input, bert_output, mask = self.forward_feature(input, wt_mask_ratio, bert_mask_ratio)
            return wt_conv_output, bert_input, bert_output, mask
        elif self.mode == 'to_finetune':
            mamba_output = self.forward_feature(input, wt_mask_ratio, bert_mask_ratio)
            cls_output = mamba_output[:, -1]
            classify_output = self.forward_head(cls_output)
            if self.confidence:
                confidence = self.confidence_generator(cls_output)
                return classify_output, confidence
            return classify_output
        else:
            # wtconv_output = self.forward_feature_only_wtconv(input, wt_mask_ratio, bert_mask_ratio)
            # classify_output = self.forward_head(wtconv_output)
            # return classify_output

            # evaluate
            mamba_output = self.forward_feature(input, wt_mask_ratio, bert_mask_ratio)
            cls_output = mamba_output[:, -1]
            if self.confidence:
                confidence = self.confidence_generator(cls_output)
                exclude_index_list = []
                index_list = []

                # print("confidence", confidence)
                for i in range(len(confidence)):
                    if confidence[i] < self.confidence:     # 不放进去，直接给标签的样本
                        exclude_index_list.append(i)
                    else:
                        index_list.append(i)                # 可以直接放进去的样本
                
            index_list = torch.tensor(index_list) 
            exclude_index_list = torch.tensor(exclude_index_list)
            cls_output_remain = cls_output[index_list]

            classify_output = self.forward_head(cls_output_remain)

            if self.confidence:
                return classify_output, confidence, index_list, exclude_index_list
            return classify_output



def randow_mask_mamba(mask_ratio, x):
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
