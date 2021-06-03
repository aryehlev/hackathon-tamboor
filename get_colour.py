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
    return 1 / abs(colour - other_colour)

def sum_w_s(colour, array_of_other_colours):
    
    return np.sum(map(lambda a: w(colour, a),array_of_other_colours ))

# def shepards_interpolation():

#     pass


if __name__ == '__main__':
    print(sum_w_s([3],[4]))