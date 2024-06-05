import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLSTMWithAttention(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers=1):
        super(ResNetLSTMWithAttention, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, sequence_length, c, h, w = x.size()

        # ResNet feature extraction
        c_out = []
        for t in range(sequence_length):
            with torch.no_grad():
                c_out.append(self.pool(self.resnet(x[:, t, :, :, :])).view(batch_size, -1))

        c_out = torch.stack(c_out, dim=1)

        # LSTM feature extraction
        lstm_out, _ = self.lstm(c_out)

        # Attention mechanism
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = torch.softmax(attn_weights, dim=1)

        attn_applied = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out).squeeze(1)

        # Classification
        out = self.fc(attn_applied)

        return out


# Define the model, loss function, and optimizer
model = ResNetLSTMWithAttention(hidden_size=256, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
