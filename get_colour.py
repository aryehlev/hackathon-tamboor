import numpy as np
from math import dist
from functools import reduce


def get_colour():
    pass


# distance between two arrays(numpy or regular) of colours
def distance(first_colour, second_colour):
    return dist(first_colour, second_colour)

#1 / distance of two numbers(colours one channel)
def w(colour, other_colour):
    return np.divide(1, np.absolute(np.subtract(colour, other_colour)))
    

def w_s(colour, array_of_other_colours):
    return map(lambda a: w(colour, a),array_of_other_colours)

def sum_w_s_c(colour, array_of_other_colours, distances):
    return np.dot(w_s(colour, array_of_other_colours), distances)

def distances_ref_real(original_4_closest, ref_4_closest):
    return np.subtract(original_4_closest, ref_4_closest)


def shepards_interpolation(original_4_closest: np.numarray, ref_4_closest: np.numarray, colour):
    distances = distances_ref_real(original_4_closest, ref_4_closest)

    sum_w_c = sum_w_s_c(colour, original_4_closest, distances) 

    sum_w = np.sum(w_s)

    return np.divide(sum_w_c, sum_w)


# if __name__ == '__main__':
#     print(sum_w_s([3],[4]))