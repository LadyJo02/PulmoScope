import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, functional

def create_deep_classifier(hidden_dim, n_classes, dropout):
    return nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, n_classes)
    )

class MultiScaleTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k,
                      padding=((k-1)//2)*dilation,
                      dilation=dilation)
            for k in [3,5,7]
        ])
        self.bn = nn.BatchNorm1d(out_ch*3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch*3, 1)

    def forward(self, x):
        y = torch.cat([c(x) for c in self.convs], dim=1)
        return self.dropout(self.relu(self.bn(y))) + self.res(x)

class Model_Pure_TCN(nn.Module):
    def __init__(self, input_channels=264, hidden_dim=192, dropout=0.15, n_classes=4):
        super().__init__()
        self.tcn1 = MultiScaleTCNBlock(input_channels, 64, 1, dropout)
        self.tcn2 = MultiScaleTCNBlock(192, 64, 2, dropout)
        self.tcn3 = MultiScaleTCNBlock(192, 64, 4, dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.classifier = create_deep_classifier(hidden_dim, n_classes, dropout)

    def forward(self, x):
        x = self.tcn3(self.tcn2(self.tcn1(x)))
        x = x.transpose(1,2)
        w = torch.softmax(self.attn(torch.tanh(x)), dim=1)
        return self.classifier((x*w).sum(dim=1))

class Model_TCN_SNN(nn.Module):
    def __init__(self, input_channels=264, hidden_dim=192, dropout=0.2, n_classes=4):
        super().__init__()
        self.tcn1 = MultiScaleTCNBlock(input_channels, 64, 1, dropout)
        self.tcn2 = MultiScaleTCNBlock(192, 64, 2, dropout)
        self.tcn3 = MultiScaleTCNBlock(192, 64, 4, dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.neuron = neuron.ParametricLIFNode(init_tau=2.0,
                         surrogate_function=surrogate.ATan())
        self.attn = nn.Linear(hidden_dim, 1)
        self.classifier = create_deep_classifier(hidden_dim, n_classes, dropout)

    def forward(self, x):
        functional.reset_net(self)
        x = self.bn(self.tcn3(self.tcn2(self.tcn1(x))))
        x_t = x.permute(2,0,1)
        spk = torch.stack([self.neuron(x_t[t]) for t in range(x_t.size(0))])
        spk = spk.permute(1,2,0)
        w = torch.softmax(self.attn(torch.tanh(spk.transpose(1,2))), dim=1)
        return self.classifier((spk.transpose(1,2)*w).sum(dim=1))
