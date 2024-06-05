import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet_lstm_attention import ResNetLSTMWithAttention
from dataset import InjectionDataset  # Assuming this file contains the dataset definition

# Hyperparameters
batch_size = 4

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = InjectionDataset(root_dir='injection-dataset_student', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = ResNetLSTMWithAttention(hidden_size=256, num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
