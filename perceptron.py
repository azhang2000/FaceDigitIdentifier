import numpy as np
import os
import sys
import math
import random as rr
import time
# load images in
# identify features


def getLabels(path):
    f = open(path, 'r')
    labels = np.loadtxt(path, delimiter=' ').astype(int)
    # print(labels)
    return labels


def getLabelData(labels, numClasses, labelSize):
    count = []  # count the number of each classes that occur in training data
    for i in range(numClasses):
        count.append(0)

    for i, x in enumerate(labels):  # counts number of elements in each class
        if i >= labelSize:
            break
        count[x] += 1

    return count  # returns a list containing the number of each different label


def countTotalLabels(count):  # counts total number of labels in training data
    total = 0
    for x in count:
        total += x
    return total


# gets the probability that a label is in a given class
def getClassProb(count, total, numClasses):
    probabilities = []  # probabilities to be returned

    for i in range(numClasses):
        probabilities.append(0)

    for x in range(numClasses):
        probabilities[x] = count[x] / total
    return probabilities


def readImages(path, imageType):
    f = open(path)
    lines = f.readlines()  # Create a list containing all lines
    f.close()
    # print(lines)
    imageList = []  # stores all images from file
    image = ''  # stores temporarily as we add valid data line by line for that image

    lineno = 1  # keeps count of what line we are on while saving an image
    if imageType == "f":
        for line in lines:
            line = line.replace("\n", "")
            image = image + line
            if lineno % 70 == 0:
                imageList.append(image)
                image = ''
            lineno = lineno + 1

    elif imageType == "d":
        lineno = 0
        while lineno < len(lines):
            line = lines[lineno]
            if ("#" in line or "+" in line):
                i = 0
                while(i < 20):
                    line = line.replace("\n", "")
                    image = image + line
                    lineno += 1
                    i += 1
                    line = lines[lineno]
                imageList.append(image)
                image = ""
                lineno = lineno - 1
            lineno += 1

    return imageList


def determineMax(YgivenX, label, currMax):
    # print('NEW Y|X LABEL', label)
    # print('current max', currMax[0])
    # print('potential new max', YgivenX)
    if YgivenX > currMax[0]:
        currMax = [YgivenX, label]

    return currMax


def getFeatures(image):
    features = [1]  # w_0 this has no feature to pair
    for item in image:
        if item == " ":
            features.append(1)
        if item == "#":
            features.append(2)
        if item == "+":
            features.append(3)
    return features


def calculateF(f, features):
    sol = 0

    for x, fx in enumerate(features):
        sol += f[x] * fx
    return sol


def getY(f):
    if f >= 0:
        return 1
    return 0


def updateF(F, features, y):
    sign = 0
    if y == False:
        sign = -1
    if y == True:
        sign = 1
    ##F[0]+= 1 * sign
    for i in range(len(F)):
        F[i] += (sign*features[i])
    return F


def getTrainingFeatures(images):
    trainingFeatures = []
    for image in images:
        trainingFeatures.append(getFeatures(image))
    return trainingFeatures


def trainPerceptron(images, labels, featureSize, numClasses, trainingSize):

    allF = []
    for x in range(numClasses):
        f = [1]
        for y in range(featureSize):  # hard coded this would need to change for digits
            # initialize f(x_i,w_i) with weights in (-1,1)
            f.append(np.random.uniform(-1, 1))

        allF.append(f)

    updates = 1
    time = 0

    trainingFeatures = getTrainingFeatures(images)
    while updates:
        updates = 0
        index = 0
        # for every example on our training set
        for i, features in enumerate(trainingFeatures):
            if i >= trainingSize:
                break
            y_i = labels[index]  # ground truth for training image label
            # list of F vals for each set of weights, <0 is false, >0 is true, index max value is the guess
            guesses = np.zeros(numClasses)
            for classNo, f in enumerate(allF):
                F = calculateF(allF[classNo], features)
                guesses[classNo] = F
            # print(guesses)
            y = np.argmax(guesses)  # find max of guess
            # print(y)
            if guesses[y] > 0 and y == y_i:  # check to see if it matches with y_i
                index += 1
                continue  # if true, continue

            if guesses[y] > 0 and y != y_i:  # we incorrectly guessed
                # update for the incorrect value guessed
                allF[y] = updateF(allF[y], features, False)
            if guesses[y_i] < 0:
                # update the value that should have been guessed if it is less than 0
                allF[y_i] = updateF(allF[y_i], features, True)

            updates += 1

            index += 1  # index of labels

        time += 1
        print(time)
        if time > 25:
            return allF
    return allF


def writeTime(time):
    fileName = 'p' + sys.argv[1] + 'Time.txt'
    f = open(fileName, 'a')
    time = str(time) + '\n'
    f.write(time)
    f.close()
    print(time)


def perceptron(trainingImages, testImages, trainingLabels, testLabels, numClasses, trainingSize):
    featureSize = len(trainingImages[0])
    # print(featureSize)
    start = time.time()

    allWeights = trainPerceptron(
        trainingImages, trainingLabels, featureSize, numClasses, trainingSize)

    end = time.time()

    writeTime(end-start)

    results = []
    for testImage in testImages:
        fx = getFeatures(testImage)
        guesses = []
        for weights in allWeights:
            fxiw = calculateF(weights, fx)
            guesses.append(fxiw)

        sol = np.argmax(guesses)
        results.append(sol)

    return results


def test(results, truth):
    numTestLabels = len(truth)
    correct = 0
    count = 0
    for t in truth:
        if t == results[count]:
            correct += 1
        count += 1

    print(correct/numTestLabels)
    # having trouble writing float to a text file
    return str(correct/numTestLabels) + ', '
    # could just do everything in this program


if __name__ == "__main__":
    # digit or face recognition
    if sys.argv[1] == 'd':  # digit recognition
        dataType = 'digit'
        # needed because of file names differ for digits and faces
        pathHelper = 'trainingimages'
        numClasses = 10
    if sys.argv[1] == 'f':  # face recognition
        dataType = 'face'
        # needed because of file names differ for digits and faces
        pathHelper = dataType + 'datatrain'
        numClasses = 2

    if sys.argv[2] == 'm':  # mac paths
        imagePath = 'data/' + dataType + 'data/' + pathHelper
        labelPath = 'data/' + dataType + 'data/' + pathHelper + 'labels'
        testPath = 'data/' + dataType + 'data/' + dataType + 'datatest'

    if sys.argv[2] == 'w':  # windows paths
        imagePath = 'data\\' + dataType + 'data\\' + pathHelper
        labelPath = 'data\\' + dataType + 'data\\' + pathHelper + 'labels'
        testPath = 'data\\' + dataType + 'data\\' + dataType + 'datatest'

    labels = getLabels(labelPath)  # list of every training label
    trainingImages = readImages(imagePath, sys.argv[1])

    c = list(zip(trainingImages, labels))
    rr.shuffle(c)
    trainingImages, labels = zip(*c)

    trainingLabelSize = int(int(sys.argv[3])/100 * len(labels))

    count = getLabelData(labels, numClasses, trainingLabelSize)

    print('****Labels occur with the following frequency****')
    n = 0
    for x in count:
        print(n, ': ', x)
        n += 1

    print('\nTraining Label size: ', trainingLabelSize)

    probabilities = getClassProb(count, trainingLabelSize, numClasses)
    #print("\nClass probabilities", probabilities)

    # read training images into objects or array

    trainingSize = int(int(sys.argv[3]) / 100 * len(trainingImages))

    print("Number of training images used", trainingSize)

    testImages = readImages(testPath, sys.argv[1])
    print("Number of test images ", len(testImages))

    testLabelPath = "data/facedata/facedatatestlabels"
    if sys.argv[1] == "d":
        testLabelPath = 'data/digitdata/digitdatatestlabels'
    testLabels = getLabels(testLabelPath)

    results = perceptron(trainingImages, testImages, labels,
                         testLabels, numClasses, trainingSize)
    print(results)

    fileName = 'resultsPerceptron.txt'
    f = open(fileName, 'a')
    f.write(test(results, testLabels))
    f.close()
