from sklearn.model_selection import train_test_split
import xlrd
import numpy as np


def read_data(data="kaggle_data.xlsx", n = 0):
    resArray = []
    data = xlrd.open_workbook(data)
    table = data.sheet_by_index(0)
    for i in range(table.nrows):
        line = table.row_values(i)
        resArray.append(line)
    x = np.array(resArray)
    onedata = []
    X = []
    y = []
    i = 0
    print(len(resArray))
    while i < len(resArray) :
        bool = 0
        for n in range(10):
            if bool == 1:
                i += 1
                continue
            elif i == 0:
                onedata.append(resArray[i][:-1])
            elif resArray[i][-1] != resArray[i-1][-1]:
                bool = 1
            else:
                onedata.append(resArray[i][:-1])
            i = i + 1
            if i >= len(resArray): break
        print(i)

        if bool == 0:
            X.append(onedata)
            y.append(resArray[i-1][-1])

    print(y)

    #
    #
    #
    #
    # for i in range(len(x)):
    #     # for num in range(len(x[i][:-1])):
    #     #     x[i][num] = float(x[i][num])
    #     for n in range(10):
    #         if x[i][-1] == x[i-1][-1]:
    #             X.append(onedata)
    #         else:
    #             break
    #
    #
    #     onedata.append(x[i][:-1])
    #     if (i + 1) % 10 == 0 and x[i][-1] == x[i-1][-1]:
    #         X.append(onedata)
    #         y.append(int(x[i][-1]))
    #     elif (i + 1) % 10 == 0:
    #         onedata = []

    X = np.array(X)
    X = X.astype(float)
    print(X.shape)

    return X, y


X, y = read_data()

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
#
# print(X) # (6823, 80)
# print(y)
#
# Xtrain = np.array(Xtrain)
# y = np.array(y) # (6823,)
#
# Xtrain = np.concatenate(list(Xtrain)).astype(np.float32)
# Xtrain = torch.tensor(Xtrain)
# print(Xtrain.shape)