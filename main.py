import torch
from torchviz import make_dot
from resnet_lstm_attention import ResNetLSTMWithAttention  # Assuming this file contains the model definition

model = ResNetLSTMWithAttention(hidden_size=256, num_classes=2)
x = torch.randn(1, 8, 3, 224, 224)  # Example input tensor
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'pdf'
dot.render('model_structure')
