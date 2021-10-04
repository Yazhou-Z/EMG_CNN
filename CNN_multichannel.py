import xlrd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable


def read_data(data="kaggle_data.xlsx", n = 0):
    resArray = []
    data = xlrd.open_workbook(data)
    table = data.sheet_by_index(0)
    for i in range(table.nrows):
        line = table.row_values(i)
        resArray.append(line)
    x = np.array(resArray)
    X = []
    y = []

    for i in range(len(x)):
        for num in range(len(x[i][:-1])):
            x[i][num] = float(x[i][num])
        X.append(x[i][:-1])
        y.append(x[i][-1])

    X = np.array(X)
    X = X.astype(float)

    return X, y


X, y = read_data()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

print(X)
print(y)

X = np.array(X) # (6823, 80)
y = np.array(y) # (6823,)


class Cnn1d(nn.Module):

    def __init__(self, in_channels, out_channels, n_len_seg, n_classes, device, verbose=False):
        super(Cnn1d, self).__init__()
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.verbose = verbose

        # 6823, 80
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=3, stride=3, padding=2),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True))
        # 2275, 80
        self.conv2 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3))
        # 759, 80
        self.conv3 = nn.Sequential(
            nn.Conv1d(80, 160, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3))
        # 253, 160
        self.conv4 = nn.Sequential(
            nn.Conv1d(160, 160, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3))
        # 85, 160
        self.conv5 = nn.Sequential(
            nn.Conv1d(160, 320, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 29, 320
        self.conv6 = nn.Sequential(
            nn.Conv1d(320, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 10, 320
        self.conv7 = nn.Sequential(
            nn.Conv1d(320, 640, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3)
        )
        # 4, 320
        self.conv8 = nn.Sequential(
            nn.Conv1d(320, 640, kernel_size=4, stride=1),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4)
        )
        # 1, 640
        self.conv9 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # 1, 640
        self.fc = nn.Linear(640, 80)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1, -1)
        # x : 6823, 80

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        # logit = self.activation(logit)

        return logit


batch_size = 1
learning_rate = 0.0001
num_epoches = 50

if torch.cuda.is_available():
    print('cuda')
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
epoch = 0
while epoch < num_epoches:
    for i in range(len(Xtrain)):
        datas = Xtrain[i]
        label = Ytrain[i]
        if torch.cuda.is_available():
            datas = datas.cuda()
            label = label.cuda()

        out = model(datas)
        out = torch.unsqueeze(out, 0)

        label = torch.tensor(label, dtype=torch.long)
        label = torch.unsqueeze(label, 0)

        loss = torch.nn.CrossEntropyLoss()(out, label)

        data = [datas, label]
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch: {}, loss: {:.4}'.format(epoch, print_loss), 'step: ', i + 1)

    epoch += 1
    if epoch % 10 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, print_loss))

# test
model.eval()
eval_loss = 0
eval_acc = 0
for i in range(len(Xtest)):
    datas = Xtest[i]
    label = Ytest[i]
    if torch.cuda.is_available():
        datas = datas.cuda()
        label = label.cuda()
    out = model(datas)
    out = torch.unsqueeze(out, 0)

    label = torch.tensor(label, dtype=torch.long)
    label = torch.unsqueeze(label, 0)
    loss = torch.nn.CrossEntropyLoss()(out, label)
    data = [datas, label]

    eval_loss += loss*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(Xtest)),
    eval_acc / (len(Xtest))
))