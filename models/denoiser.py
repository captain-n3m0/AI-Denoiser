import torch
import torchaudio
import torch.nn as nn

class DenoiserModel(nn.Module):
    def __init__(self):
        super(DenoiserModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model():
    model = DenoiserModel()
    model.load_state_dict(torch.load("models/pretrained_model.pth"))
    model.eval()
    return model
