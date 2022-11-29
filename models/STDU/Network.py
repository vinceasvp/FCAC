import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
from models.resnet18_encoder import resnet18
from models.resnet20_cifar import resnet20

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)

        self.mode = mode
        self.args = args
        self.encoder = resnet18(True, args)  # pretrained=False
        self.num_features = 512
        self.fc = nn.Linear(self.num_features, self.args.num_all, bias=False)
        hdim=self.num_features
        self.beta = 1.0
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.transatt_proto = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        if args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
            self.set_fea_extractor_for_s2s()
        else:
            self.set_module_for_audio(args)
            
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

    def _forward(self, support, query, pqa=False, sup_emb=None, novel_ids=None):  # support and query are 4-d tensor, shape(num_batch, 1, num_proto, emb_dim)
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
        if sup_emb is not None:
            att_proto = self.get_att_proto(sup_emb, query, num_query, emb_dim)
            mean_proto_expand.data[:, :, novel_ids, :] = self.beta * att_proto.unsqueeze(0) \
                                                    + (1-self.beta) * mean_proto_expand[:, :, novel_ids, :]
        proto = mean_proto_expand.view(num_batch*num_query, num_proto, emb_dim)

        if pqa:
            combined = torch.cat([proto, query], 1)  # Nk x (N + 1) or (N + 1 + 1) x d, batch_size = NK
            combined, _ = self.slf_attn(combined, combined, combined)
            proto, query = combined.split(num_proto, dim=1)
        else:
            combined = proto
            combined, _ = self.slf_attn(combined, combined, combined)
            proto = combined

        logits=F.cosine_similarity(query, proto, dim=-1)
        logits=logits*self.args.network.temperature
        return logits, anchor_loss, query, proto
        
    def get_att_proto_shot_score(self, sup_emb, num_query, emb_dim):
        sup_emb = sup_emb.view(self.args.episode.episode_shot, -1, emb_dim).permute(1, 0, 2)
        att_emb, att_logit = self.inneratt_proto(sup_emb, sup_emb, sup_emb)
        # att_proto = att_emb.mean(dim=1)

        shot_logit = att_logit.mean(dim=1)
        shot_score = F.softmax(shot_logit, dim=1)
        shot_score = shot_score.unsqueeze(-1)
        att_proto = shot_score * sup_emb
        att_proto = att_proto.sum(1)
        att_proto_expand = att_proto.unsqueeze(0).expand(num_query, -1, emb_dim)
        return att_proto_expand

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

    def set_module_for_audio(self, args):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
            win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
            n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

        # speechbrain tools 
        self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                            win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                            hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                            n_fft=self.args.extractor.window_size)
        self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)
    
    def set_fea_extractor_for_s2s(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(128) 

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        assert len(data) == self.args.episode.episode_way * self.args.episode.episode_shot
        
        if not self.args.strategy.data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.network.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.network.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.network.new_mode:
            return self.args.network.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        num_base = self.args.stdu.num_tmpb if self.args.tmp_train else self.args.num_base
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs.epochs_new):
                old_fc = self.fc.weight[:num_base + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[num_base + self.args.way * (session - 1):num_base + self.args.way * session, :].copy_(new_fc.data)
 
    def encode(self, x):
        """
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        """
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
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
