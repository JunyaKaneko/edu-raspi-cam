# Copyright 2021 Morning Project Samurai, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
# A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import numpy as np
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import click


class PersonDataset(Dataset):
    def __init__(self, person_dir, bg_dir, transform=None):
        self._dataset = [(os.path.join(person_dir, f), 1) for f in os.listdir(person_dir)] \
                        + [(os.path.join(bg_dir, f), 0) for f in os.listdir(bg_dir)]
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        path, label = self._dataset[item]
        data = Image.open(path)
        if self._transform:
            data = self._transform(data)
        return data, label


class Head(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_features, n_classes, bias=True)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


def downsample():
    return nn.MaxPool2d(2, 2)


class Features(nn.Module):
    def __init__(self, n_features):
        super(Features, self).__init__()
        self.features = nn.Sequential(conv3x3(3, 8),
                                      downsample(),
                                      conv3x3(8, 8),
                                      downsample(),
                                      conv3x3(8, 16),
                                      downsample(),
                                      conv3x3(16, 32),
                                      downsample(),
                                      conv3x3(32, 64),
                                      conv3x3(64, n_features))

    def forward(self, x):
        return self.features(x)


def predict(model, image, transform, device):
    x = transform(image).unsqueeze(0).to(device)
    y = nn.functional.softmax(model(x), dim=-1)[0].to(device)
    feature_map = model[0](x)[0]
    weight = model[1].fc.weight
    return y, feature_map, weight


def create_cam(feature_map, weight, y, y_th, device):
    n_feature_map = feature_map.shape[0]
    activation_map = torch.zeros(feature_map.shape[1:]).to(device)
    for i in range(n_feature_map):
        if y[1] < y_th:
            continue
        activation_map += weight[1, i] * feature_map[i]
    return torch.clip(activation_map * 255. / (torch.max(activation_map) + 1e-20), 0)


def draw_cam(image, cam, out_path):
    im_cam = Image.fromarray(np.uint8(cam.tolist())) \
        .resize(image.size) \
        .convert('RGB')
    im_cam = im_cam.resize(image.size).convert('RGB')
    im_cam = ImageChops.multiply(image, im_cam)
    im_cam.save(out_path)


@click.group()
def cli():
    pass


@cli.command()
def train():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter((0.8, 1.0), (0.8, 1.0), (0.8, 1.0)),
        torchvision.transforms.ToTensor(),
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = PersonDataset('../data/train/person', 'data/train/background', train_transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

    val_dataset = PersonDataset('../data/val/person', 'data/val/background', val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(Features(64), Head(64, 2))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for e in range(50):
        running_loss = 0.
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('[%d] loss: %.5f' % (e + 1, running_loss / len(train_dataset)))

    with torch.no_grad():
        error = 0.
        for i, data in enumerate(val_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            error += torch.sum(torch.abs(predicted - labels))
        print(1. - error / len(val_dataset))

    model.cpu()
    torch.save(model.state_dict(), '../models/model.pth')


@cli.command()
@click.option('--pth')
@click.option('--th', default=0.5)
@click.option('--ipt')
@click.option('--opt')
@click.option('--workdir', default='/tmp')
def eval(pth, th, ipt, opt, workdir):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(Features(64), Head(64, 2))
    model.load_state_dict(torch.load(pth))
    model.to(device)
    model.eval()

    cam_lock = os.path.join(workdir, 'cam_lock')
    while True:
        if not os.path.exists(ipt):
            continue
        try:
            x = Image.open(ipt).convert('RGB')
        except OSError:
            continue
        y, feature_map, weight = predict(model, x, transform, device)
        cam = create_cam(feature_map, weight, y, th, device)
        open(cam_lock, 'w').close()
        draw_cam(x, cam, opt)
        os.remove(cam_lock)


if __name__ == '__main__':
    cli()
