import pandas as pd


def get_closest_colours(guessed_colour):
    all_colours_df = pd.read_excel('RGB Color fan.xlsx')

    # guessed_colour_as_string = f"{guessed_colour[0]};{guessed_colour[1]};{guessed_colour[2]}"

    all_colours_df['RGB'] = all_colours_df['RGB'].apply(lambda x: [int(a) for a in x.split(';')])

    all_colours.insert(location, column_name, list_of_values)

    closest_colours = all_colours_df.sort_values(by=["RGB"], key=sorter)