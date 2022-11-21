import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)

        self.args = args
        hdim=self.num_features
        self.beta = 1.0
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.transatt_proto = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode == 'fm_encoder':
            if self.args.dataset == "fsdclips":
                x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
                x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                # x = x.transpose(1, 3)
                # x = self.bn0(x)
                # x = x.transpose(1, 3)
                x = x.repeat(1, 3, 1, 1)
            input = self.encoder(input)
            return input
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits

    def _forward(self, support, query, transductive=False, sup_emb=None, novel_ids=None):  # support and query are 4-d tensor, shape(num_batch, 1, num_proto, emb_dim)
        anchor_loss = 0.0
        emb_dim = support.size(-1)
        num_query = query.shape[1]*query.shape[2]#num of query*way
        query = query.view(-1, emb_dim).unsqueeze(1)  # shape(num_query, 1, emb_dim)

        # get mean of the support of shape(batch_size, shot, way, dim)
        mean_proto = support.mean(dim=1, keepdim=True)  # calculate the mean of each class's prototype without keeping the dim
        num_batch = mean_proto.shape[0]
        num_proto = mean_proto.shape[2]  # num_proto = num of support class

        # the shape of proto is different from query, so make them same by coping (num_proto, emb_dim)
        mean_proto_expand = mean_proto.expand(num_batch, num_query, num_proto, emb_dim).contiguous()  # can be regard as copying num_query(int) proto
        shortcut_mean_proto_expand = mean_proto_expand.view(num_batch*num_query, num_proto, emb_dim)
        if sup_emb is not None:
            if self.args.STDU.ap.ap_type == "outer":
                att_proto = self.get_att_proto(sup_emb, query, num_query, emb_dim)
            mean_proto_expand.data[:, :, novel_ids, :] = self.beta * att_proto.unsqueeze(0) \
                                                    + (1-self.beta) * mean_proto_expand[:, :, novel_ids, :]
        proto = mean_proto_expand.view(num_batch*num_query, num_proto, emb_dim)

        combined = torch.cat([proto, query], 1)  # Nk x (N + 1) or (N + 1 + 1) x d, batch_size = NK
        combined, _ = self.slf_attn(combined, combined, combined)

        logits=F.cosine_similarity(query, proto, dim=-1)
        logits=logits*self.args.network.temperature
        return logits, anchor_loss, query, proto
        

    def get_att_proto(self, sup_emb, query, num_query, emb_dim):
        sup_emb = sup_emb.unsqueeze(0).expand(num_query, sup_emb.shape[0], sup_emb.shape[-1])
        cat_emb = torch.cat([sup_emb, query], dim=1)
        att_pq, att_logit = self.transatt_proto(cat_emb, cat_emb, cat_emb)
        att_logit = att_logit[:, :, -1][:, :-1] # 选取最后一列的前shot*way个logits
        att_score = torch.softmax(att_logit.view(num_query, self.args.episode.episode_shot, -1), dim=1)
        att_proto, _ = att_pq.split(sup_emb.shape[1], dim=1)
        att_proto = att_proto.view(num_query, self.args.episode.episode_shot, -1, emb_dim) * att_score.unsqueeze(-1) # self.args.episode_way+self.args.low_way
        att_proto = att_proto.sum(1)
        return att_proto

    def get_featmap(self, input):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        feat_map = self.encoder(input)
        return feat_map

    def pre_encode(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn_logit = torch.bmm(q, k.transpose(1, 2))
        attn = attn_logit / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn_logit, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn_logit, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn_logit


if __name__ == "__main__":
    proto = torch.randn(25, 512, 2, 4)
    query = torch.randn(75, 512, 2, 4)
