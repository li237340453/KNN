from numpy import shape, tile
from numpy.matlib import zeros

from classify0 import classify0


def file2matrix(filename):
    fr = open(filename)
    arrayOLine = fr.readlines()
    numberOfLines = len(arrayOLine)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLine:
        line = line.strip()
        listFromline = line.split('\t')
        returnMat[index, :] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('E:\code\PyCharmProject\kNN\match\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(classifierResult, "    ", datingLabels[i])
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print('\n')
    print(errorCount / float(numTestVecs))


if __name__ == '__main__':
    datingClassTest()
