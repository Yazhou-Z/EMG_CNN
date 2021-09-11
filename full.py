import xlrd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim


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
        X.append(x[i][:-1])
        y.append(int(x[i][-1]))

    X = np.array(X)
    y = np.array(y)

    return X, y


path = 'bu_data_for_ML.xlsx'
X, y = load_excel(path)

print(X.shape)
y = np.array(y)
print(y)
print(X)
#X = StandardScaler().fit_transform(X)
#print(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


batch_size = 64
learning_rate = 0.02
num_epoches = 200

model = Net(106*1, 300, 100, 2)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
epoch = 0
for i in range(len(X)):
    datas = Xtrain[i]
    label = Ytrain[i]
    if torch.cuda.is_available():
        datas = datas.cuda()
        label = label.cuda()
    out = model(datas)
    print(label)

    #label = np.reshape(label, 1)

    label = label.reshape(label, 1)

    print(label.shape)
    label = torch.tensor(label)

    label = label.to(torch.float32)
    print(label.shape)

    print(out)
    print(out.shape)
    # loss = criterion(out, label)
    loss = torch.nn.CrossEntropyLoss()(label, out)
    data = [datas, label]
    # data = np.array(datas, label)
    # print(data.np.shape)
    print_loss = loss.data.item()
    # print_loss = torch.nn.CrossEntropyLoss()(label, out)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

# test
model.eval()
eval_loss = 0
eval_acc = 0
for i in range(len(X)):
    datas = Xtest[i]
    label = Ytest[i]
    if torch.cuda.is_available():
        datas = datas.cuda()
        label = label.cuda()
    data = [datas, label]

    out = model(datas)
    loss = criterion(out, label)
    eval_loss += loss.data.item()*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(Xtest)),
    eval_acc / (len(Xtest))
))
