import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import time


def mean(lst):
    return sum(lst)/len(lst)


if __name__ == "__main__":
    start = time.time()
    accuracy = [0.755, 0.79, 0.758, 0.739,
                0.743, 0.775, 0.751, 0.747, 0.769, 0.784]

    mean = stat.mean(accuracy)
    print('Mean = ', mean)

    std = stat.stdev(accuracy)
    print('Std =', std)

    end = time.time()

    print(end-start)
    # need mean and standard deviation for algorithms naive:faces and digits, perceptron: faces and digits
