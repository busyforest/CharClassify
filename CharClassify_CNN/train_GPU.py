import datetime

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets
from torchvision import transforms

from Le_Net5_model import LeNet5

data_transform = transforms.Compose([
    transforms.Grayscale(),  # 将图像转换为灰度值
    transforms.ToTensor()    # 将图像转换为张量
])

train_set_path = r"train_data_CNN/train"
data_train = datasets.ImageFolder(train_set_path, transform=data_transform)
data_loader = Data.DataLoader(data_train, batch_size=64, shuffle=True)

model = LeNet5()
Epoch = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("正在使用：", torch.cuda.get_device_name())
batch_size = 64
lr = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch.set_grad_enabled(True)
model.train()
start_time = datetime.datetime.now()
for epoch in range(Epoch):
    running_loss = 0.0
    acc = 0.0
    for step, data in enumerate(data_loader):
        x, y = data
        optimizer.zero_grad()
        y_pred = model(x.to(device, torch.float))
        loss = loss_function(y_pred, y.to(device, torch.long))
        loss.backward()
        running_loss += float(loss.data.cpu())
        pred = y_pred.argmax(dim=1)
        acc += (pred.data.cpu() == y.data).sum()
        optimizer.step()
        if step % 100 == 0:
            loss_avg = running_loss / (step + 1)
            acc_avg = acc / ((step + 1) * batch_size)
            print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg, '|Acc_avg:%.4f' % acc_avg)
# torch.save(model, './model.pkl')
end_time = datetime.datetime.now()
print("训练时间：", end_time - start_time)