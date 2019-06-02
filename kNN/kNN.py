from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet() :
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ["A","A","B","B"]
    return group, lables


def classif0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat **2
    spDistances = sqDiffMat.sum(axis = 1)
    distances = spDistances ** 0.5
    print("Distance vector is ", distances)
    sortedDisIndicies = distances.argsort()
    print("Sorted indict is ", sortedDisIndicies)
    classCount = {}
    for i in range(k) :
        voteILable = lables[sortedDisIndicies[i]]
        classCount[voteILable] = classCount.get(voteILable,0) + 1
    print("Class Count is ", classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

group, lables = createDataSet()
print("kNN result is ", classif0([0,0], group, lables, 3))

fig = plt.figure()
ax = fig.add_subplot(222)
ax.scatter([1,1,0, 0],[1.1,1.0,0,0.1],[100,100,50,50],[100,100,50,50] )
plt.show()


