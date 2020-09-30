import numpy as np
import os
from statistics import mean
import sys


def toFloat(lst):
    newList = []
    for i, x in enumerate(lst):
        newList[i] = float(x)
    return newList


def average(lst):
    return mean(lst)


def formatting(algo, percentage):
    print('formatting')
    fileName = algo + 'Time.txt'
    f = open(fileName, 'a')
    format = '****' + str(percentage) + '\n'
    f.write(format)
    f.close()


if __name__ == "__main__":
    iterations = 10
    percentage = 10
    if sys.argv[1] == 'p':  # run perceptron a few times for each percentage
        if sys.argv[2] == 'f':
            algo = sys.argv[1] + sys.argv[2]

            while percentage < 101:  # iterates through percentages
                fileName = 'resultsPerceptron.txt'
                f = open(fileName, 'a')
                f.write(str(percentage))
                f.write('****Faces\n')
                f.close()
                # run 10%, 20%,..,100%
                # collect statistics by running program at percentage
                formatting(algo, percentage)

                for x in range(iterations):
                    command = 'Python perceptron.py f m ' + str(percentage)
                    os.system(command)
                percentage += 10  # increment %

        if sys.argv[2] == 'd':  # run perceptron a few times for each percentage
            while percentage < 101:  # iterates through percentages
                fileName = 'resultsPerceptron.txt'
                f = open(fileName, 'a')
                f.write(str(percentage))
                f.write('****Digits\n')
                f.close()
                # run 10%, 20%,..,100%
                # collect statistics by running program at percentage
                for x in range(iterations):
                    command = 'Python perceptron.py d m ' + str(percentage)
                    os.system(command)
                percentage += 10  # increment %

    if sys.argv[1] == 'n':  # run perceptron a few times for each percentage
        if sys.argv[2] == 'f':
            while percentage < 101:  # iterates through percentages
                fileName = 'resultsNaive.txt'
                f = open(fileName, 'a')
                f.write(str(percentage))
                f.write('****Faces\n')
                f.close()
                # run 10%, 20%,..,100%
                # collect statistics by running program at percentage
                for x in range(iterations):
                    command = 'Python naive.py f m ' + str(percentage)
                    os.system(command)
                percentage += 10  # increment %

        if sys.argv[2] == 'd':  # run perceptron a few times for each percentage
            while percentage < 101:  # iterates through percentages
                fileName = 'resultsNaive.txt'
                f = open(fileName, 'a')
                f.write(str(percentage))
                f.write('****Digits\n')
                f.close()
                # run 10%, 20%,..,100%
                # collect statistics by running program at percentage
                for x in range(iterations):
                    command = 'Python naive.py d m ' + str(percentage)
                    os.system(command)
                percentage += 10  # increment %

    # run perceptron n times
    # find the average accuracy
    # standard deviation, variance
    # min and max accuracy
