import pandas as pd # pyright: reportMissingImports=false
import get_colour
from get_colour import distance_two_colours 

def get_closest_colours(guessed_colour):
    all_colours_df = pd.read_excel('RGB Color fan.xlsx')

    all_colours_df['RGB'] = all_colours_df['RGB'].apply(lambda x: [int(a) for a in x.split(';')])

    all_colours_df.insert(0, 'distances', all_colours_df['RGB'].apply(lambda x: distance_two_colours(guessed_colour, x)))

    closest_colours = all_colours_df.sort_values(by=["distances"])

    return closest_colours.head(4)

if __name__ == '__main__':
    print(get_closest_colours([112,112,112])['RGB'])