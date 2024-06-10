import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import InjectionDataset
from resnet_lstm_attention import ResNetLSTMWithAttention

# 设备配置：使用GPU加速如果可用，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
model = ResNetLSTMWithAttention(hidden_size=256, num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()  # 设置模型为评估模式

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = InjectionDataset(root_dir='injection-dataset_student', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 测试循环
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
