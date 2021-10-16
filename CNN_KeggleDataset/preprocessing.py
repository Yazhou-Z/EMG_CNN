from sklearn.model_selection import train_test_split
import xlrd
import numpy as np
import torch

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
    print(len(resArray))
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

    print(yy)
    X = np.array(X)
    X = X.astype(float)
    print(X.shape)
    print(len(y))

    return X, y



X, y = read_data()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

print(X.shape)    # (630, 10, 80), C: 80, N: 631, L: 10
print(y)    # 630

Xtrain = np.array(Xtrain)
y = np.array(y) # (6823,)

Xtrain = np.concatenate(list(Xtrain)).astype(np.float32)
Xtrain = torch.tensor(Xtrain)
print(Xtrain.shape)
