from sklearn import tree
import pandas as pd
import numpy as np
import os

debug = 0

"""
This function takes in a site_name, and a list of csv_names and
returns a dataframe that can be used with
the decision tree model designed later.
"""
def create_dataframe(csv_names:list[str], site_number:int):
    types_of_protocols = ['TCP', 'TLSv1.3', 'TLSv1.2']
    minute_data = []

    def compute_minute(df):
        #reset and declare variables
        num_of_TCP = 0
        num_of_TLSv1_2 = 0
        num_of_TLSv1_3 = 0

        len_of_TCP = 0
        len_of_TLSv1_2 = 0
        len_of_TLSv1_3 = 0
        
        for index, row in df.iterrows():    
            #catch protocol and use to calculate the relevant data
            if row['Protocol'] == 'TCP':
                num_of_TCP += 1
                len_of_TCP += row['Length']

            elif row['Protocol'] == 'TLSv1.2':
                num_of_TLSv1_2 += 1
                len_of_TLSv1_2 += row['Length']

            elif row['Protocol'] == 'TLSv1.3':
                num_of_TLSv1_3 += 1
                len_of_TLSv1_3 += row['Length']
        minute_data = [num_of_TCP, num_of_TLSv1_2, num_of_TLSv1_3, len_of_TCP, len_of_TLSv1_2, len_of_TLSv1_3, site_number]    
        return minute_data

    #this code is not as relevant, it just computes the same minute number_of_minutes times to test the dataframe
    
    total_data = []
    if debug > 0:
        print(csv_names)
    for name in csv_names:
        if debug > 0:
            print(name[:-4])
        if (name[-4:] == '.csv'):
            df = pd.read_csv(name)
            minute_data = compute_minute(df)
            total_data.append(minute_data)

    # Put all the data into a dataframe so it's easier to use!
    dataframe = pd.DataFrame(total_data, columns = ['Number of TCP Packets', 
                                                    'Number of TLSv1.2 Packets', 
                                                    'Number of TLSv1.3 Packets',
                                                    'Total Length of TCP Packets',
                                                    'Total Length of TLSv1.2 Packets',
                                                    'Total Length of TLSv1.3 Packets',
                                                    'Website'])
    return dataframe