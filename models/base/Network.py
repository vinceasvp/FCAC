import argparse
from speechbrain.processing.features import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset == 'FMC':
            self.encoder = resnet18(True, args)  # pretrained=False
            self.num_features = 512
        if 'nsynth' in self.args.dataset:
            self.encoder = resnet18(True, args)  # pretrained=False
            self.num_features = 512
        if 'librispeech' in self.args.dataset:
            self.encoder = resnet18(True, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
            self.encoder = resnet18(True, args)  # pretrained=False
            self.num_features = 512
        self.fc = nn.Linear(self.num_features, self.args.num_all, bias=False)
        self.set_module_for_audio(args)

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

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.network.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.network.temperature * x
        return x

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

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        assert len(data) == self.args.episode.episode_way * self.args.episode.episode_shot
        
        if self.args.strategy.not_data_init:
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
        num_base = self.args.sis.num_tmpb if self.args.tmp_train else self.args.num_base
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

