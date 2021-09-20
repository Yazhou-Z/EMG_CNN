import xlrd
import numpy as np
import openpyxl

def load_excel_normalize(path):
    resArray = []
    data = xlrd.open_workbook(path)
    table = data.sheet_by_index(0)
    for i in range(table.ncols):
        line = table.col_values(i)
        resArray.append(line)
    x = np.array(resArray)
    X = []

    for i in range(len(x)):
        newL = []
        for l in range(len(x[i]) - 1):
            if x[i][l] == '':
                continue
            # print(x[i][l])
            newL.append(int(float(x[i][l])))
        # sklearn.preprocessing.normalize(newL, norm='l1', axis=1, copy=True, return_norm=False)
        m1 = np.full((len(newL)), max(newL) / 2)
        m2 = np.full((len(newL)), 1)
        normalized_L = newL / m1 - m2
        # m2 = np.full(len(normalized_L), max(normalized_L) - min(normalized_L))
        # normalized_L = newL / m2
        normalized_L = normalized_L.tolist()
        print(normalized_L)
        X.append(normalized_L)

    # X = np.array(X)
    # X = X.astype(float)
    return X


def write_excel(name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'A'
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row = j + 1, column = i + 1, value = value[i][j])
    workbook.save(name)


path = 'bu_seg_Sept20.xlsx'
X = load_excel_normalize(path)

# X = np.array(X)
print(X)

write_excel('normalized_data_Spet20_7.xlsx', X)
