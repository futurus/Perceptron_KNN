__author__ = 'vunguyen'
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


def breakWFpair(pair):
    pair = pair.split(':')
    return tuple([pair[0], int(pair[1])])


def formatLine(line):
    components = line.split(' ')

    docClass = int(components[0])
    pairs = map(breakWFpair, components[1:])

    return [docClass, dict(pairs)]


def prepare(finput):
    import operator

    lines = map(formatLine, [line for line in open(finput)])
    docClassLabels = [line[0] for line in lines]
    docs = [line[1] for line in lines]  # each doc is a dictionary of word:frequency
    dictionary = list(set(reduce(operator.add, [line[1].keys() for line in lines])))

    featureMat = zeros((len(docClassLabels), len(dictionary)))

    for i in range(len(lines)):
        for word in docs[i].keys():
            # bag of word
            featureMat[i, dictionary.index(word)] = docs[i][word]

            # set of word
            # if docs[i][word] > 0:
            #     featureMat[i, dictionary.index(word)] = 1
            # else:
            #     featureMat[i, dictionary.index(word)] = 0


    return dictionary, docClassLabels, featureMat  # return matrix and labels


def translateDoc(finput, dictionary):
    lines = map(formatLine, [line for line in open(finput)])
    docClassLabels = [line[0] for line in lines]
    docs = [line[1] for line in lines]  # each doc is a dictionary of word:frequency

    featureMat = zeros((len(docClassLabels), len(dictionary)))

    for i in range(len(lines)):
        for word in docs[i].keys():
            if word in dictionary:
                featureMat[i, dictionary.index(word)] = docs[i][word]

    return docClassLabels, featureMat


def getPriorDists(labels):
    priorDists = {}

    for i in range(len(set(labels))):
        priorDists[i] = 0

    for i in range(len(labels)):
        priorDists[labels[i]] += 1

    for key in priorDists.keys():
        priorDists[key] /= float(len(labels));

    return priorDists


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
        print 'Class', i, 'accuracy:', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy:', overallAccuracy(actual, prediction)
    return None


def logNovern(data, dictionary):
    print "logNovern: start"
    toc()
    numOfWords = len(dictionary)
    print "dictionary size:", numOfWords
    out = zeros(numOfWords)
    for i in range(numOfWords):
        out[i] = math.log(1. * numOfWords / sum(data[:, i]))

    print "logNovern: stop"
    toc()
    return out


def tfidf(matrix, lNovern):
    print "tfidf: start"
    toc()
    out = zeros((len(matrix), len(matrix[0])))

    for i in range(len(out)):
        out[i] = matrix[i]/math.sqrt(sum(matrix[i]**2)) * lNovern

    print "tfidf: start"
    toc()
    return out


def similarity(sample, base):
    normX = math.sqrt(sum(sample**2))
    normD = math.sqrt(sum(base**2))
    denominator = normX * normD

    return sum(sample * base) / denominator


def vote(neighbors, labels):
    tally = dict([(c, 0) for c in set(labels)])

    for i in neighbors:
        tally[labels[i]] += 1

    return list(OrderedDict(sorted(tally.items(), key=lambda x: x[1], reverse=True)))[0]  # sort desc


def knnClassify(k=10):
    dictionary, trainLabels, trainData = prepare("p2/8category.training.txt")
    testLabels, testData = translateDoc("p2/8category.testing.txt", dictionary)

    lNovern = logNovern(trainData, dictionary)
    W = tfidf(trainData, lNovern)  # tfidf matrix of training data
    T = tfidf(testData, lNovern)   # tfidf matrix of test data
    print "data import completed"
    toc()

    predictions = []

    for i in range(len(testLabels)):  # going through test cases
        print "instance:", i
        nearestNeighbors = {}
        for j in range(len(trainLabels)):
            nearestNeighbors[j] = similarity(T[i], W[j])

        nearestNeighbors = OrderedDict(sorted(nearestNeighbors.items(), key=lambda x: x[1], reverse=True))  # sort desc
        predictions.append(vote(list(nearestNeighbors)[:k], trainLabels))
        toc()

    analyze(testLabels, predictions)


tic()
knnClassify(k=12)
toc()