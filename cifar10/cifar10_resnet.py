# Data: https://www.kaggle.com/c/cifar-10/data

# %%
import ipyenv as uu
uu.chdir(__file__)
dir_data = "data/cifar10"

# %% 一次失败的尝试
from torchvision.datasets import ImageFolder
# dataset = ImageFolder(dir_data+'/train', transform=ToTensor())

# %% 加载数据
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import ToTensor  # ToPILImage

import os
import pandas as pd
import numpy as np
import torch, torchvision
from torchvision.datasets.folder import default_loader  # pil_loader

class ImageFolder2(data.Dataset):
    def __init__(self, dir_root, path_label, transform, target_transform=None, loader=default_loader):
        self.root = os.path.expanduser(dir_root)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        # self.images = os.listdir(path_dir)
        labels = pd.read_csv(path_label)
        targets = self._build_classes(labels)
        self.samples = []
        for tuple_row in targets.itertuples():
            # img_id = self.labels.iat[index,0]
            # target = self.labels.iat[index,1]
            _idx, img_id, target = tuple_row
            path_img = os.path.join(self.root, f"{img_id}.png")
            self.samples.append((path_img, target))
        self.imgs = self.samples

    def _build_classes(self, labels):
        self.classes = labels["label"].unique()
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        # self.idx_to_class = {i:c for c, i in self.class_to_idx.items()}
        labels["label"] = labels["label"].map(self.class_to_idx)
        return labels

    def __getitem__(self, index):
        path_img, target = self.samples[index]
        sample = self.loader(path_img)

        if self.transform:
            sample = self.transform(sample)
        else:
            # im_tensor = transforms.ToTensor()(img)  # 归一化
            sample = torch.from_numpy(np.asarray(sample))
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])  # 标准化至[-1,1]
])

datasets = ImageFolder2(dir_data+"/train",
                        dir_data+"/trainLabels.csv",
                        transform)

# %% 测试: 显示图像
import matplotlib.pyplot as plt

def show_image(data_item):
    img, target = data_item
    print('Label: ', datasets.classes[target])
    plt.imshow(img.permute(1,2,0))

show_image(datasets[0])

# %% 测试加载
train_loader = data.DataLoader(datasets, batch_size=4, shuffle=True)

inputs, labels = next(iter(train_loader))
print(inputs.shape, labels)

# %% 划分: 训练集-测试集
def train_test_split(datasets, test_size=None, train_size=None, random_state=0):
    if (not test_size) and train_size:
        test_size = 1-train_size
    assert test_size, "未定义的test_size划分比例"
    _total = len(datasets)
    n_test = int(_total * test_size)
    n_train = _total - n_test

    args = [datasets, [n_train, n_test]]
    if random_state:
        args.append(torch.manual_seed(random_state))

    return data.random_split(*args)

train_ds, val_ds = train_test_split(datasets, test_size = 0.1)
print(len(train_ds), len(val_ds), type(val_ds))

# %%
batch_size = 128
train_dl = data.DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)  # num_workers=4
val_dl = data.DataLoader(val_ds, batch_size, pin_memory=True)  # num_workers=4

# %% 显示图像
def show_images_batch(dataloader):
    for images, _labels in dataloader:
        _fig, ax= plt.subplots(figsize=(16,8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(torchvision.utils.make_grid(images, nrow=16).permute(1,2,0))
        break  # to stop loop otherwise 4500 images in batch size of 128 will print and is computationally expensive

show_images_batch(train_dl)

# %%
import torch.nn as nn
import torch.nn.functional as F
load_pretrained = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.resnet18(pretrained=load_pretrained)
model = torchvision.models.resnet50(pretrained=load_pretrained)
if load_pretrained:
    model.load_state_dict(torch.load(path_model))

model.to(device)  # model into cuda
# print(model)

# %% 修改模型
# 原本的输出：(fc): Linear(in_features=512, out_features=1000, bias=True)
inchannel = model.fc.in_features  # 重新定义fc全连接层，此时，会进行参数的更新
model.fc = nn.Linear(inchannel, len(datasets.classes))
# print(model)

# %%
if load_pretrained:
    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model.parameters():
        param.requires_grad = False  # 冻结参数的更新
    # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
    for param in model.fc.parameters():
        param.requires_grad = True

# 建议在硬件条件允许的情况下，可以不使用这个固定参数的模型，对所有层的参数
# 进行学习的精确性肯定更好，resnet本身这个模型的参数不是很庞大

# %%
from pytorchtools import EarlyStopping

def fit(model, train_loader, loss_func, optimizer, grad_clip=None):
    model.train()  # 设置模型到训练模式，只对部分模型有影响（包含Dropout，BatchNorm）
    for images, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(images.to(device))
        loss = loss_func(output, labels.to(device))
        loss.backward()

        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        print(f'Loss: {loss.item():.6f}')

@torch.no_grad()
def evaluate(model, val_loader, loss_func):
    model.eval()  # 设置模型到测试模式
    val_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += loss_func(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        # accuracy = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct += torch.sum(pred == target).item()

    val_size = len(val_loader.dataset)
    val_loss /= val_size
    print(f'Test-Loss: {val_loss:.4f}, Accuracy: {correct}/{val_size} ({100*correct/val_size:.0f}%)')
    return val_loss


# Set up cutom optimizer with weight decay
epochs = 5
path_model = uu.rpath("tmp/cifar10_resnet.pt")
path_checkpoint = uu.rpath("tmp/checkpoint.pt")
early_stopping = EarlyStopping(patience=7, verbose=True, path=path_checkpoint)

torch.cuda.empty_cache()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Set up one-cycle learning rate scheduler
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epochs,
                                            steps_per_epoch=len(train_dl))
try:
    for epoch in range(epochs):
        fit(model, train_dl, F.cross_entropy, optimizer)
        val_loss = evaluate(model, val_dl, F.cross_entropy)
        if early_stopping(val_loss, model):
            model.load_state_dict(torch.load(early_stopping.path))
            break
        sched.step()  # update learning rate
except Exception as e:
    print(f">>> 中断: {e}\n保存模型至【{path_model}】")
finally:
    torch.save(model.state_dict(), path_model)

# %% 加载模型，恢复运算
if False:
    net = torchvision.models.resnet18(pretrained=False)
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, len(datasets.classes))

    net.load_state_dict(torch.load(path_model))

    # evaluate(net, val_dl, F.cross_entropy)
