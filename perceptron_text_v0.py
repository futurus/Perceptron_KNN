__author__ = 'vunguyen'
from numpy import *
import time


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


def alpha(t):  # learning rate
    return 60.0 / (59.0 + t)


def shuffle(order):
    random.shuffle(order)
    return order


def trainPerceptrons(epoch=1000, bias=False, randomize=False):
    dictionary, trainLabels, trainData = prepare("p2/8category.training.txt")

    numOfTrainInstances = len(trainLabels)
    numOfFeatures = len(trainData[0])
    numOfClasses = len(set(trainLabels))

    if bias:
        numOfFeatures += 1
        trainData = insert(trainData, 0, 1, axis=1)

    if randomize:
        W = random.rand(epoch, numOfClasses, numOfFeatures)
    else:
        W = zeros((epoch, numOfClasses, numOfFeatures))

    for episode in range(1, epoch):
        misclassification = ones(numOfTrainInstances)  # helps keeping track of which instances need adjustments
        W[episode] = W[episode-1]

        for instance in range(numOfTrainInstances):
            tempResult = []
            for c in range(numOfClasses):
                tempResult.append(sum(W[episode, c] * trainData[instance]))

            bestguess = tempResult.index(max(tempResult))
            truevalue = trainLabels[instance]

            if bestguess != truevalue:  # if we guess wrong
                W[episode, bestguess] = W[episode, bestguess] - alpha(episode) * trainData[instance]
                W[episode, truevalue] = W[episode, truevalue] + alpha(episode) * trainData[instance]
            else:
                misclassification[instance] = 0

        print "i:", episode, ", accuracy:", 1-sum(misclassification)/numOfTrainInstances

    return W[epoch-1], numOfClasses, dictionary


def classify(epoch=200, bias=False, randomize=False):
    predictions = []
    print "start training"
    W, numOfClasses, dictionary = trainPerceptrons(epoch=epoch, bias=bias, randomize=randomize)
    toc()
    print "done training"

    testLabels, testData = translateDoc("p2/8category.testing.txt", dictionary)

    if bias:
        testData = insert(testData, 0, 1, axis=1)

    for instance in range(len(testLabels)):
        tempResult = []
        for c in range(numOfClasses):
            tempResult.append(sum(W[c] * testData[instance]))

        predictions.append(tempResult.index(max(tempResult)))

    analyze(testLabels, predictions)


tic()
classify(epoch=15, bias=False, randomize=False)
toc()