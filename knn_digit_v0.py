__author__ = 'vunguyen'
dimension = 28
from numpy import *
import math
import time
from collections import OrderedDict


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def replace(char):
    if char is ' ':
        return 0
    else:
        return 1


def prepare(finput):
    out = []
    lines = [line for line in open(finput)]

    for nDigit in range(len(lines)/dimension):
        digit = []
        for i in range(dimension):
            cline = list(lines[nDigit*dimension + i])
            cline.pop()  # remove new line char
            cline = map(replace, cline)
            digit.append(cline)
        out.append(digit)

    return array(out)


def getRaw(finput, case):
    lines = [line for line in open(finput)]

    for i in range(dimension):
        print lines[case * dimension + i][:(dimension-1)]


def getLabels(finput):
    return [int(line[0]) for line in open(finput)]


def getPriorDists(labels):
    priorDists = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for i in range(len(labels)):
        priorDists[labels[i]] += 1

    for key in priorDists.keys():
        priorDists[key] /= float(len(labels));

    return priorDists


def bin2dec(a):
    return int(''.join(map(str, map(int, a.flatten()))), 2)


def digitToFeatures(digit, m=1, n=1, overlap=False):  # by default we look at a pixel
    features = []

    if m > digit.shape[0] or n > digit.shape[1]:
        print "m or n too large (max 28)"
        return None

    if overlap is False and ((28 % m != 0) or (28 % n != 0)):
        print "m or n is not a divisor of 28"
        return None

    if overlap is False:
        for row in range(0, digit.shape[0], m):
            for col in range(0, digit.shape[1], n):
                features.append(bin2dec(digit[row:(row+m), col:(col+n)]))
    else:
        for row in range(digit.shape[0]-m+1):
            for col in range(digit.shape[1]-n+1):
                features.append(bin2dec(digit[row:(row+m), col:(col+n)]))

    return features


def getData(digits, m=1, n=1, overlap=False):
    trainData = []

    for i in range(len(digits)):
        trainData.append(digitToFeatures(digits[i], m, n, overlap))

    return array(trainData)


def confusionMatrix(actual, prediction):
    mat = zeros((len(set(actual)), len(set(actual))))

    for i in range(len(actual)):
        mat[actual[i], prediction[i]] += 1

    return mat


def overallAccuracy(actual, prediction):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            correct += 1

    return correct*1.0/len(actual)


def analyze(actual, prediction):
    confMat = confusionMatrix(actual, prediction)

    print confMat

    for i in range(len(confMat)):
        print 'Digit', i, 'accuracy:', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy:', overallAccuracy(actual, prediction)
    return None


def similarity(sample, base):
    return sum((sample - base) ** 2)


def vote(neighbors, labels):
    tally = dict([(c, 0) for c in set(labels)])

    for i in neighbors:
        tally[labels[i]] += 1

    return list(OrderedDict(sorted(tally.items(), key=lambda x: x[1], reverse=True)))[0]  # sort desc


def knnClassify(k=10):
    trainData = getData(prepare("p1/trainingimages"))
    trainLabels = getLabels("p1/traininglabels")
    numOfTrainInstances = len(trainLabels)
    numOfFeatures = len(trainData[0])
    numOfClasses = 10
    testData = getData(prepare("p1/testimages"))
    testLabels = getLabels("p1/testlabels")
    print "data import completed"
    toc()

    predictions = []

    for i in range(len(testLabels)):  # going through test cases
        nearestNeighbors = {}
        for j in range(len(trainLabels)):
            # time.sleep(2)
            nearestNeighbors[j] = similarity(testData[i], trainData[j])

        nearestNeighbors = OrderedDict(sorted(nearestNeighbors.items(), key=lambda x: x[1], reverse=False))  # sort asc
        predictions.append(vote(list(nearestNeighbors)[:k], trainLabels))

    analyze(testLabels, predictions)


tic()
knnClassify(k=10)
toc()