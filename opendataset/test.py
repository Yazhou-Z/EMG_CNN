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
    i = 0
    yy = 0
    # print(len(resArray))
    while i < len(resArray):
        bool = 0
        onedata = []
        for n in range(10):
            if bool == 1:
                i += 1
                continue
            elif i == 0:
                yy += 1
                onedata.append(list(resArray[i][:-1]))
            elif resArray[i][-1] != resArray[i-1][-1]:
                bool = 1
            else:
                onedata.append(list(resArray[i][:-1]))
            i = i + 1
            if i >= len(resArray):
                bool = 1
                break

        if bool == 0:
            onedata = np.array(onedata)
            print(onedata.shape)
            onedata = np.transpose(onedata) # reshape (630, 10, 80): (N, L, C) to (N, C, L)
            X.append(onedata)
            y.append(resArray[i-1][-1])

    # print(yy)
    X = np.array(X)
    X = X.astype(float)
    # print(X.shape)
    # print(len(y))

    return X, y


X, y = read_data("kaggle_data.xlsx")

X = torch.tensor(X)
y = torch.tensor(y)
X_tensor = X.view(630, 80, 10)
y = y.view(630, 1)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

print(X.shape)    # torch.Size([630, 80, 10]) N, C, L
print(y.shape)    # torch.Size([630, 1])

# Xtrain = np.array(Xtrain)
# Xtrain = np.concatenate(list(Xtrain)).astype(np.float32)
# Xtrain = torch.tensor(Xtrain)
# print(Xtrain.shape)


class Cnn1d(nn.Module):

    def __init__(self,
                 num_class = 7
                 # in_size, out_channels,
                 # # n_len_seg,
                 # n_classes,
                 # # device,
                 # verbose=False
                 ):
        super(Cnn1d, self).__init__()
        # self.n_len_seg = n_len_seg
        # self.n_classes = n_classes
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.device = device
        # self.verbose = verbose

        # input: (N, C, L) (630, 80, 10)

        # (630, 80, 10)
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, 81, kernel_size=2, stride=1, padding=1), # 81, 10
            nn.BatchNorm1d(81),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2))  # 81, 5
        # (630, 81, 5)
        self.conv2 = nn.Sequential(
            nn.Conv1d(81, 27, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(27),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1)
        )  # 27, 2
        self.conv3 = nn.Sequential(
            nn.Conv1d(27, 9, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLU(inplace=True))
        # 9, 2
        self.conv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(18, 7)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        print("input: ", x.shape)   # torch.Size([1, 80, 10])
        out = self.conv1(x)
        print("conv1:", out.shape)  # torch.Size([1, 80, 10])
        out = self.conv2(out)
        print("conv2", out.shape)
        out = self.conv3(out)
        print("conv3", out.shape)
        out = self.conv4(out)
        print("conv4", out.shape)
        # out = out.view(out.size(0), -1)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        logit = self.activation(logit)
        print("fc:", logit.shape)

        return logit
        # return out


batch_size = 1
learning_rate = 0.0001
num_epoches = 5

model = Cnn1d(7)

# if torch.cuda.is_available():
#     print('cuda')
#     model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
epoch = 0
while epoch < num_epoches:
    for i in range(len(X_tensor)):
        datas = X_tensor[i]
        datas = datas.to(torch.float32)
        datas = datas.unsqueeze(0)
        print(datas.shape)  # torch.Size([1, 80, 10])
        if torch.cuda.is_available():
            datas = datas.cuda()
            label = label.cuda()

        out = model(datas)
        print(out)
        out = torch.unsqueeze(out, 0)

        label = y[i]
        # label = label.view(7, 1)
        print(label.shape)  # torch.Size([1])
        label = torch.tensor(label, dtype=torch.long)
        label = torch.unsqueeze(label,1)
        print("label: ", label.shape)  # torch.Size([1])

        # loss = criterion(out, label)
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

# # test
# model.eval()
# eval_loss = 0
# eval_acc = 0
# for i in range(len(Xtest)):
#     datas = Xtest[i]
#     label = Ytest[i]
#     if torch.cuda.is_available():
#         datas = datas.cuda()
#         label = label.cuda()

#     datas = torch.tensor(datas, dtype=torch.long)

#     out = model(datas)
#     out = torch.unsqueeze(out, 0)

#     label = torch.tensor(label, dtype=torch.long)
#     label = torch.unsqueeze(label, 0)
#     loss = torch.nn.CrossEntropyLoss()(out, label)
#     data = [datas, label]

#     eval_loss += loss*label.size(0)
#     _, pred = torch.max(out, 1)
#     num_correct = (pred == label).sum()
#     eval_acc += num_correct.item()
# print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
#     eval_loss / (len(Xtest)),
#     eval_acc / (len(Xtest))
# ))
