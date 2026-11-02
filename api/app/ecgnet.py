import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        s.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(cout)
        s.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False);     s.b2 = nn.BatchNorm1d(cout)
        s.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        s.act = nn.ReLU()

    def forward(s, x):
        o = s.act(s.b1(s.c1(x)))
        o = s.b2(s.c2(o))
        return s.act(o + s.sk(x))


class ECGNet(nn.Module):
    """CNN 1D résiduel — identique à la partie D du rapport (entrée : 12 × 1000)."""

    def __init__(s):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(12, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        s.layers = nn.Sequential(Block(32, 64, 2), Block(64, 64, 1), Block(64, 128, 2),
                                 Block(128, 128, 1), Block(128, 256, 2), Block(256, 256, 1))
        s.drop = nn.Dropout(0.5)
        s.fc = nn.Linear(512, 5)

    def forward(s, x):
        x = s.layers(s.stem(x))
        x = torch.cat([x.mean(-1), x.amax(-1)], 1)
        return s.fc(s.drop(x))
