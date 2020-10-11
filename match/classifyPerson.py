from pip._vendor.distlib.compat import raw_input
from numpy import *

from classify0 import classify0
from match.resolve import file2matrix, autoNorm


def classifyPerson():
    resultList = ['not in all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("video games"))
    ffMiles = float(raw_input("filter miles"))
    iceCream = float(raw_input("ice cream"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print(resultList[classifierResult-1])


if __name__ == '__main__':
    classifyPerson()
