# -*- coding: utf-8 -*-
from network import Network
from activate_functions import Logistic
from layer import FullyConnectedLayer
from backpropagation import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# RANDOM_SEED = 42

#Загрузка данных из файла
def loadDataFromFile(path):
    f = open(path, "rb")
    data = []
    labels = []
    for str in f:
        substr = str.split(",")
        tmp = []
        for i in range(0, 9):
            if substr[i] == "o":
                tmp.append(1)
            if substr[i] == "x":
                tmp.append(-1)
            if substr[i] == "b":
                tmp.append(0)
        data.append(tmp)
        if substr[9][:len(substr[9])-1] == "negative":
            labels.append(0)
        else:
            labels.append(1)
    print len(data)
    data = np.array(data)
    labels = np.array(labels)
    return train_test_split(data, labels, test_size=0.33)

#построение кривой ошибок
def plot(error_curve):
    plt.plot([x for x in range(0, len(error_curve))], error_curve)
    plt.show()

#тестирование
def testing(net, data, labels):
    output = net.activate(data)
    answer = (output > 0.5)
    answer = answer.reshape(len(answer))
    percentage = (answer == labels).sum() / float(data.shape[0])  * 100
    return percentage

#вычисление показателей ROC
def calcROC(net, data, labels):
    output = net.activate(data)
    answer = output > 0.5
    answer = answer.reshape(len(answer))
    TP = TN = FP = FN = 0
    for i in range(0, len(answer)):
        if answer[i] == labels[i] == 1:
            TP += 1
        if answer[i] == labels[i] == 0:
            TN += 1
        if answer[i] == 1 and labels[i] == 0:
            FP += 1
        if answer[i] == 0 and labels[i] == 1:
            FN += 1
    print 'TP = ' +  str(TP)
    print 'TN = ' + str(TN)
    print 'FP = ' + str(FP)
    print 'FN = ' + str(FN)
    Precision = float(TP) / (TP + FP)
    Sensitivity = float(TP) / (TP + FN)
    print 'Sensitivity = ' + str(float(TP) / (TP + FN))
    print 'Specificity = ' + str(float(TN) / (TN + FP))
    print 'Precision = ' + str(float(TP) / (TP + FP))
    print 'F-score = ' + str(2 * (Precision * Sensitivity)/(Precision + Sensitivity))

#построение ROC-кривой
def drawROCCurve(net, data, labels):
    output = net.activate(data)
    answer = output.reshape(len(output))
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    t = 0
    tmax = 1
    dx = 0.0001
    points = []
    while t <= tmax:
        FP = TP = 0
        for i in range(0, len(answer)):
            if answer[i] >= t:
                if labels[i] == 1:
                    TP += 1
                else:
                    FP += 1
        SE = TP / float(P)
        m_Sp = FP / float(N)
        points.append([m_Sp, SE])
        t += dx
    print points
    points.reverse()
    points = np.array(points)
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(points[:, 0], points[:, 1], lw=2, label='ROC curve')
    plt.plot([0.0, 1.0], [0.0, 1.0], lw=2)
    plt.show()
    auc = 0
    for i in xrange(1, len(points)):
        auc += (points[i, 0] - points[i - 1, 0]) * points[i, 1]
    print 'auc = ' + str(auc)


#загрузка данных из файла
data = loadDataFromFile("Datasets/tic-tac-toe.data.txt")
#конфигурирование сети
net = Network()
layer_1 = FullyConnectedLayer(Logistic(), 9, 9)
layer_3 = FullyConnectedLayer(Logistic(), 9, 1)
net.append_layer(layer_1)
net.append_layer(layer_3)
params = Backprop_params(500, 1e-5, 10, 0.9, False, [0.01, 0.01], 0)
method = Backpropagation(params, net)
train_data = data[0]
test_data = data[1]
train_labels = data[2]
test_labels = data[3]

#обучение
error_curve = method.train(train_data, train_labels)
plot(error_curve)
#вывод результатов
print "Train efficiency: " + str(testing(net, train_data, train_labels))
print "Test efficiency: " +  str(testing(net, test_data, test_labels))

#вычисление показателей ROC
calcROC(net, test_data, test_labels)
drawROCCurve(net, test_data, test_labels)


