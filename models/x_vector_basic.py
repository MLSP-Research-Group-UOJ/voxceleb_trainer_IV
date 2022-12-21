#! /usr/bin/python
# -*- encoding: utf-8 -*-

import pdb
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import weight_norm


class MainModel(nn.Module):
    def __init__(self, nOut=1024, n_mels=None, log_input=None, p=0.5, **kwargs):
        super(MainModel, self).__init__();

        self.log_input = log_input

        print('Embedding size is %d, %d mel filterbanks.' % (nOut, n_mels))

        self.netcnn = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=p),

            nn.Conv1d(512, 512, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=p),

            nn.Conv1d(512, 512, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=p),

            nn.Conv1d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=p),

            nn.Conv1d(512, 1500, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(1500),
            nn.Dropout(p=p),
        );

        self.fc = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=p),

            nn.Linear(512, nOut),
        );

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                            hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=n_mels)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        x = self.torchfb(x) + 1e-6
        if self.log_input: x = x.log()
        x = self.instancenorm(x).detach()

        x = self.netcnn(x);

        x = torch.cat((torch.mean(x, dim=2), torch.sqrt(torch.var(x, dim=2).clamp(min=1e-5))), dim=1)

        x = self.fc(x);

        return x;

