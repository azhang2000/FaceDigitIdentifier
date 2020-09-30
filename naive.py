import numpy as np
import os
import sys
import math
import random as rr
import time
# load images in
# identify features


class Number:
    def __init__(self, value, image):
        self.value = value
        self.image = image


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


def initializeDataStore():
    return


def checkResults(guesses, dataType):
    if dataType == "f":
        truth = getLabels('data/facedata/facedatatestlabels')
    else:
        truth = getLabels('data/digitdata/digitdatatestlabels')
    correct = 0
    count = 0
    for t in truth:
        if t == guesses[count]:
            correct += 1
        count += 1
    return(correct/len(truth))


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


# performs a summary of all the training data based off of the label and character at each pixel
def getTrainingData(trainingImages, trainingLabels, numClasses, dataType, trainingSize):
    # creates a 3D matrix  [label][charactertype][index of image] which stores the total number of #, +, or spaces at that location

    picSize = 4200
    if dataType == "d":
        picSize = 560

    allTrainingData = np.zeros((numClasses, 3, picSize))

    # loops through all training images
    for imageNo, trainingImage in enumerate(trainingImages):
        if imageNo >= trainingSize:
            break
        # loops through a single training image
        for index, item in enumerate(trainingImage):
            if item == " ":
                allTrainingData[trainingLabels[imageNo]][0][index] += 1
            elif item == "#":
                allTrainingData[trainingLabels[imageNo]][1][index] += 1
            else:
                allTrainingData[trainingLabels[imageNo]][2][index] += 1
    # maybe try to divide by class size for each modified position

    #print('All training data', allTrainingData[0][0][200])
    return allTrainingData


def writeTime(time):
    fileName = 'n' + sys.argv[1] + 'Time.txt'
    f = open(fileName, 'a')
    time = str(time) + '\n'
    f.write(time)
    f.close()
    print(time)


def NaiveBayes(trainingImages, testImages, trainingLabels, numClasses, yProb, labelCount, dataType, trainingSize):

    start = time.time()
    trainingData = getTrainingData(
        trainingImages, trainingLabels, numClasses, dataType, trainingSize)
    end = time.time()
    writeTime(end - start)

    allProb = []  # what we determined the images to be based on the training data

    count = 0
    for testImage in testImages:  # goes through test images
        count += 1
        probability = []  # probability that an image belongs to a particular class
        classNo = 0

        while classNo < numClasses:  # cycles though the different classes to get likelihood
            probXgivenY = 0
            # finds the probability an image belongs in a class given its features
            for index, item in enumerate(testImage):
                # print(probXgivenY)
                if item == " ":
                    probXgivenY = math.log(
                        (trainingData[classNo][0][index]+1)/(labelCount[classNo]+1)) + probXgivenY
                elif item == "#":
                    probXgivenY = math.log(
                        (trainingData[classNo][1][index]+1)/(labelCount[classNo]+1)) + probXgivenY
                elif item == "+":
                    probXgivenY = math.log(
                        (trainingData[classNo][2][index]+1)/(labelCount[classNo]+1)) + probXgivenY
            classNo += 1  # move to next class
            probability.append(probXgivenY)
        # print(count)
        #print('Probability of image', count, '[not face, face]', probability)
        # print('P(X|Y)*P(Y) =', probability[0]
            # + math.log(yProb[0]), probability[1] + math.log(yProb[1]))
        classNo = 0  # reset class cycle
        maxGuess = 0
        # [probability of being in a class, label of that class]
        while classNo < numClasses:  # finds most likely prediction
            YgivenX = probability[classNo] + math.log(yProb[classNo])
            if YgivenX > probability[maxGuess]:
                maxGuess = classNo
            classNo += 1
        allProb.append(maxGuess)  # append label to the solutions
    print(allProb)
    return allProb


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
    # counts number of labels in each class
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
    print("\nClass probabilities", probabilities)

    trainingSize = int(int(sys.argv[3]) / 100 * len(trainingImages))

    print("Number of training images used", trainingSize)

    testImages = readImages(testPath, sys.argv[1])
    print("Number of test images ", len(testImages))

    results = NaiveBayes(trainingImages, testImages, labels,
                         numClasses, probabilities, count, sys.argv[1], trainingSize)

    accuracy = checkResults(results, sys.argv[1])
    print(accuracy)
    accuracy = str(accuracy) + ', '
    fileName = 'resultsNaive.txt'
    f = open(fileName, 'a')
    f.write(accuracy)
    f.close()
