import xlrd
import openpyxl
import numpy as np
from scipy import interpolate
import pylab as pl

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
            X[data][scalar] = (X[data][scalar] // spare) * spare    # + spare
            X[data][scalar] = int(X[data][scalar])
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


def write_excel(name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'A'
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row = j + 1, column = i + 1, value = int(value[i][j]))
    workbook.save(name)


# data: 一组data
# origin: 原data 有多少点
# new: 扩增后多少点
# output: 扩增后数据 y_new
def interpol(data, origin, new):
    x = np.linspace(0, origin - 1, origin)
    y = data
    pl.plot(x, y, "ro")
    x_new = np.linspace(0, origin - 1, new)
    for kind in ["quadratic", "cubic", "zero", "slinear"]:
        # kind = "quadratic"
        # kind = "cubic"
        f = interpolate.interp1d(test_x, test_y, kind=kind)
        y_new = f(x_new)
        pl.plot(x_new, y_new, label=str(kind))

    pl.legend(loc="lower right")
    pl.show()
    return y_new


path = 'bu_data_for_ML.xlsx'
X, y = load_excel(path)
print(y)

# new, label = expand_data(X, y, 5)

# test
test_y = X[0]
test_x = np.linspace(0, 105, 106)
pl.plot(test_x, test_y, "ro")

xnew = np.linspace(0, 105, 10000)

for kind in ["quadratic", "cubic", "zero", "slinear"]:
    f = interpolate.interp1d(test_x, test_y, kind=kind)
    ynew = f(xnew)
    pl.plot(xnew, ynew, label=str(kind))

pl.legend(loc="lower right")
pl.show()
# pl.savefig("interpolate.png")
new = interpol(X[0], 106, 100000000)
