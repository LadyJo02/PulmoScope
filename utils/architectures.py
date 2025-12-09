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
        nn.Linear(64, n_classes),
    )


class MultiScaleTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 5, 7],
        dilation=1,
        dropout=0.15,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                out_channels,
                k,
                padding=((k - 1) // 2) * dilation,
                dilation=dilation,
            )
            for k in kernel_sizes
        ])

        total_out = out_channels * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(total_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, total_out, 1)
            if in_channels != total_out
            else nn.Identity()
        )

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        out = self.bn(out)
        out = self.dropout(self.relu(out))
        res = self.residual(x)
        if res.size(2) != out.size(2):
            res = res[:, :, : out.size(2)]
        return out + res


class Model_Pure_TCN(nn.Module):
    def __init__(self, input_channels=264, hidden_dim=192, dropout=0.2, n_classes=4):
        super().__init__()
        self.tcn1 = MultiScaleTCNBlock(input_channels, 64, dilation=1, dropout=dropout)
        self.tcn2 = MultiScaleTCNBlock(192, 64, dilation=2, dropout=dropout)
        self.tcn3 = MultiScaleTCNBlock(192, 64, dilation=4, dropout=dropout)

        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.classifier = create_deep_classifier(hidden_dim, n_classes, dropout)

    def forward(self, x):
        x = self.tcn3(self.tcn2(self.tcn1(x)))
        x = x.transpose(1, 2)
        scores = self.attn_fc(torch.tanh(x))
        w = torch.softmax(scores, dim=1)
        return self.classifier((x * w).sum(dim=1))


class Model_TCN_SNN(nn.Module):
    def __init__(self, input_channels=264, hidden_dim=192, dropout=0.2, n_classes=4):
        super().__init__()
        self.tcn1 = MultiScaleTCNBlock(input_channels, 64, dilation=1, dropout=dropout)
        self.tcn2 = MultiScaleTCNBlock(192, 64, dilation=2, dropout=dropout)
        self.tcn3 = MultiScaleTCNBlock(192, 64, dilation=4, dropout=dropout)

        self.snn_bn = nn.BatchNorm1d(hidden_dim)
        self.neuron = neuron.ParametricLIFNode(
            init_tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.classifier = create_deep_classifier(hidden_dim, n_classes, dropout)

    def forward(self, x):
        functional.reset_net(self)

        x = self.tcn3(self.tcn2(self.tcn1(x)))
        x = self.snn_bn(x)

        x_time = x.permute(2, 0, 1)
        spk = torch.stack([self.neuron(x_time[t]) for t in range(x_time.size(0))])
        spk = spk.permute(1, 2, 0)

        attn_in = spk.transpose(1, 2)
        scores = self.attn_fc(torch.tanh(attn_in))
        w = torch.softmax(scores, dim=1)

        return self.classifier((attn_in * w).sum(dim=1))
