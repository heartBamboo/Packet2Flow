from timm.layers import NormMlpClassifierHead
from torch import nn
from models.WTConv.wtconvnext.wtconvnext import wtconvnext_tiny
from models.bert_pytorch import BERT
from utils import ClassificationHead

class wtconv_bert(nn.Module):
    """
    混合模型
    """

    def __init__(self, args,mode,hidden=256, n_layers=12, attn_heads=8):

        super().__init__()
        self.mode = mode
        self.wtconv = wtconvnext_tiny(mode,args)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bert = BERT(args,hidden, n_layers, attn_heads)
        self.pack_len=args.pack_len
        self.confidence=args.confidence
        if mode != 'to_pretrain':
            self.head = ClassificationHead(input_dim=256, num_classes=args.num_classes, dropout_rate=args.head_dropout)
            if args.confidence:
                self.confidence_generator = nn.Sequential(
                    nn.Linear(256, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                    nn.Sigmoid()
                )

    def forward_feature(self, input, wt_mask_ratio,bert_mask_ratio):
        wt_conv_output = self.wtconv(input, wt_mask_ratio)
        bert_input = self.pool(wt_conv_output).view(-1, self.pack_len, wt_conv_output.shape[1])
        if self.mode == 'to_pretrain':
            bert_output, mask = self.bert(bert_input, bert_mask_ratio)
            return wt_conv_output, bert_input, bert_output, mask
        else:
            bert_output = self.bert(bert_input, bert_mask_ratio)
            return bert_output

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
            bert_output = self.forward_feature(input, wt_mask_ratio, bert_mask_ratio)
            cls_output = bert_output[:, 0]
            classify_output = self.forward_head(cls_output)
            if self.confidence:
                confidence = self.confidence_generator(cls_output)
                return classify_output, confidence
            return classify_output
        else:
            wtconv_output = self.forward_feature_only_wtconv(input, wt_mask_ratio, bert_mask_ratio)
            classify_output = self.forward_head(wtconv_output)
            return classify_output




