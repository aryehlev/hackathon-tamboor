import numpy as np
from math import dist



def get_colour():
    pass


# distance between two arrays(numpy or regular) of colours
def distance(first_colour, second_colour):
    return dist(first_colour, second_colour)

#1 / distance of two numbers(colours one channel)
def w(first_colour, second_colour):
    return 1 / abs(first_colour - second_colour)
# def shepards_interpolation():

#     pass


if __name__ == '__main__':
    print(distance([3],[4]))