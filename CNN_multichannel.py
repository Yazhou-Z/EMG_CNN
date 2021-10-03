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
        super(CNN, self).__init__()

        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.verbose = verbose

        # 6823, 80
        self.conv1 = nn.Sequential(
            nn.Conv1d(80, 80, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        # 2275,
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 2274, 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 758, 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1 x 512
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT))
        # 1 x 512
        self.fc = nn.Linear(512, 50)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1, -1)
        # x : 23 x 1 x 59049

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        # logit = self.activation(logit)

        return logit

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        x = self.layer1(x)
        # x = F.relu(self.layer1(x))
        # x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        # x = torch.sigmoid(self.layer3(x))
        return x


batch_size = 1
learning_rate = 0.0001
num_epoches = 50

model = cnn1dNet(80, 400, 200, 50, 8)

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


'''

'''

def load_excel(path):
    resArray = []
    data = xlrd.open_workbook(path)
    table = data.sheet_by_index(0)
    for i in range(table.ncols):
        line = table.col_values(i)
        resArray.append(line)
    x = np.array(resArray)
    X = []
    y = []

    for i in range(len(x)):
        for num in range(len(x[i][:-1])):
            x[i][num] = float(x[i][num])
        X.append(x[i][:-1])
        if x[i][-1] == 1:
            y.append(1)
        else:
            y.append(0)

    X = np.array(X)
    X = X.astype(float)


    return X, y


def manage(X, spare):
    for data in range(len(X)):
        for scalar in range(len(X[data])):
            X[data][scalar] = (X[data][scalar] // spare) * spare  + spare
    return X


def expand_data(X, y, size):
    new = []
    new_label = []
    for l in range(len(X)):
        if y[l] == 0:
            label = 0
        else:
            label = 1
        for i in range(size):
            new_col = []
            for j in range(len(X[l]) // size):
                new_col.append(X[l][(j - 1) * size + i])
            new_label.append(label)
            new.append(new_col)
    new = np.array(new)
    new_label = np.array(new_label)
    return new, new_label


path = 'bu_data_for_ML.xlsx'
X, y = load_excel(path)

print(X.shape)
y = np.array(y)

X = manage(X, 10)
X, y = expand_data(X, y, 4)
X = np.array(X)
y = np.array(y)
print(X.shape)
# X = StandardScaler().fit_transform(X)
print(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        x = self.layer1(x)
        # x = F.relu(self.layer1(x))
        # x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        # x = torch.sigmoid(self.layer3(x))
        return x


batch_size = 1
learning_rate = 0.0001
num_epoches = 50

model = Net(26, 400, 200, 50, 2)

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

'''
