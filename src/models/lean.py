import torch
import torch.nn as nn

class LEAN(nn.Module):
    def __init__(self, num_classes=2):
        super(LEAN, self).__init__()

        self.lstm == nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(64 + 16 * 64 * 31, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, wave_1d, spec_2d):
        lstm_out, _ = self.lstm(wave_1d.unsqueeze(-1))
        lstm_feat = lstm_out[:, -1, :]

        cnn_feat = self.cnn(spec_2d.unsqueeze(1))

        combined = torch.cat((lstm_feat, cnn_feat), dim=1)
        x = torch.relu(self.fc1(combined))

        return self.fc2(x)
