import numpy as np
from math import dist
from functools import reduce
import sys
import pandas as pd

def get_colour():
    pass


# distance between two arrays(numpy or regular) of colours
def distance(first_colour, second_colour):
   return dist(first_colour, second_colour)

#1 / distance of two numbers(colours one channel)
def w(colour, other_colour):
    simple_dist = abs(float(colour) - float(other_colour))
    if simple_dist == 0:
        return None
    return  1 / simple_dist
    

def w_s(colour, array_of_other_colours):
    list_of_w_s = []
    
    for i in range(len(array_of_other_colours)):
        # print(1)
        # print(array_of_other_colours[i])
        curr_w = w(colour, array_of_other_colours[i])

        if not curr_w:
            return [i]

        list_of_w_s.append(curr_w)
        
    return  np.array(list_of_w_s)
    

def sum_w_s_c(colour, array_of_other_colours, distances):
    # print(array_of_other_colours)
    list_of_w_s = w_s(colour, array_of_other_colours).T
    
    return np.dot(list_of_w_s, distances)

def distances_ref_real(original_4_closest, ref_4_closest):
    return np.subtract(original_4_closest, ref_4_closest)


def shepards_interpolation(original_4_closest: np.numarray, ref_4_closest: np.numarray, colour):
    all_channels_deltas = []
    # print(original_4_closest.shape[1])
    for i in range(original_4_closest.shape[1]):
        current_chanel_original = original_4_closest[:,i]
        current_chanel_ref = ref_4_closest[:,i]
        colour_channel = colour[i]
        

        list_of_w_s = w_s(colour_channel, current_chanel_original)
        
        if (len(list_of_w_s) == 1):
            all_channels_deltas.append(current_chanel_original[list_of_w_s[0]] - current_chanel_ref[list_of_w_s[0]])
        else:
            distances = distances_ref_real(current_chanel_original, current_chanel_ref)
            sum_w_c = sum_w_s_c(colour_channel, current_chanel_original, distances) 
            sum_w = np.sum(w_s(colour_channel, current_chanel_original))

            all_channels_deltas.append(np.divide(sum_w_c, sum_w) * -1)
    print(all_channels_deltas)
    return all_channels_deltas

if __name__ == '__main__':
    shepards_interpolation(np.array([[  5 , 99 , 37],
 [  5 ,255 ,255],
 [  5 ,171  ,86],
 [  5, 171 , 86]]),np.array([[  0  ,94 , 32],
 [  0, 255 ,255],
 [  0, 166 , 81],
 [  0 ,166  ,81]]), np.array([112 ,112 ,112]))