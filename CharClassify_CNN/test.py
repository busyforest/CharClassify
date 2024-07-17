import torch
import torchvision
import torch.utils.data as Data
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度值
    transforms.ToTensor()  # 将图像转换为张量
])

test_set_path = r"train_data_CNN/test"
data_test = datasets.ImageFolder(test_set_path, transform=data_transform)
data_loader = Data.DataLoader(data_test, batch_size=64, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load('./model.pkl', map_location=torch.device(device))
net.to(device)
torch.set_grad_enabled(False)

total = 0
correct = 0
for i, data in enumerate(data_loader):
    x, y = data
    y_pred = net(x.to(device, torch.float))
    pred = y_pred.argmax(dim=1)
    correct += (pred == y.to(device, torch.float)).sum().item()
    total += y.size(0)

print('Accuracy of the network on the test images: ', 100 * correct / total, "%")
