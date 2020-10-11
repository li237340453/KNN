import numpy


def img2vector(filename):
    returnVect = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,0 + 32 * i + j] = int(lineStr[j])
        return returnVect


if __name__ == '__main__':
    print(img2vector('E:/code/PyCharmProject/kNN/number/number/0_0.txt'))