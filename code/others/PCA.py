import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def prepareData():
    irises_dataset = datasets.load_iris()
    return irises_dataset['data'], irises_dataset['target']

def __normalizing(data):
    max = data.max(axis=0)
    min = data.min(axis=0)
    data = (data - min) / (max - min)
    return data

def pca_method(data):
    cov_matrix = np.cov(data.T)
    V, PC = np.linalg.eig(cov_matrix)
    sort_index = np.argsort(-1 * V)
    PC = PC[:, sort_index]
    data = np.dot((PC.T)[0:2], data.T)
    full_info = V.sum()
    V = V[sort_index][0:2]
    reduce_info = V.sum()
    print "Reduced " + str(100 - reduce_info / full_info * 100) + " percent of information"
    return data.T

def visualise_data(data, targets):
    for i in range(0, len(data)):
        if targets[i] == 0:
            color = 'red'
        elif targets[i] == 1:
            color = 'blue'
        elif targets[i] == 2:
            color = 'green'
        plt.plot(data[i, 0], data[i, 1], 'o', color=color)
    plt.show()


if __name__ == "__main__":
    irises_data, irises_target = prepareData()
    #irises_data = __normalizing(irises_data)
    irises_reduction = pca_method(irises_data)    
    visualise_data(irises_reduction, irises_target)
    
    #pca = PCA(n_components=2)
    #X_reduced = pca.fit_transform(irises_data)
