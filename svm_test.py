from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from time import time
import numpy as np
import datetime
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import xlrd


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
        y.append(x[i][-1])

    X = np.array(X)
    y = np.array(y)

    return X, y


path = 'bu_data_for_ML.xlsx'
X, y = load_excel(path)

print(X.shape)
y = np.array(y)
print(y)
print(X)
X = StandardScaler().fit_transform(X)
print(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
kernel = "rbf"

time0 = time()
clf = SVC(kernel=kernel,
          gamma="auto",
          degree=1,
          cache_size=7000  # 允许使用多大的内存MB 默认200
          ).fit(Xtrain, Ytrain)
print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))


# rbf
# 返回在对数刻度上均匀间隔的数字
# 从-10开始取50个数到1 再把这50个数转换成对数值 默认底数为10 返回值为10的x次方
gamma_range = np.logspace(-10, 1, 50)
score = []
for i in gamma_range:
    clf = SVC(kernel="rbf", gamma=i, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), gamma_range[score.index(max(score))])
print(classification_report(Ytrain, clf.predict(Xtrain)))


def print_roc(Ytest, Xtest):
# https://blog.csdn.net/qq_45769063/article/details/106649523
    fpr, tpr, threshold = roc_curve(Ytest, clf.predict(Xtest))   # 计算真正率和假正率
    print(fpr)
    print(tpr)
    print(threshold)
    roc_auc = auc(fpr, tpr)
    plt.figure()


    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


roc_auc = print_roc(Ytest, Xtest)