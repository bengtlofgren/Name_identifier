import json
import pandas as pd
import numpy as np
import keras
from numpy import array
from numpy import argmax
import os

filenames = ['boy_names.json', 'girl_names.json', 'boy_and_girl_names.json']
name_dict = {}
for filename in filenames:
    with open(filename) as f:
        name_dict.update({filename.split('.')[0] : json.load(f)})
        
boy_names = name_dict["boy_names"]
girl_names = name_dict["girl_names"]
boy_and_girl_names = name_dict["boy_and_girl_names"]

def is_male(row, column_name):

    # In case name is first-name last-name
    name = row[column_name].split(' ')[0]
    
    if name in boy_names:
        return 1
    elif name in girl_names:
        return -1
    else:
        return 0



accepted_chars = 'abcdefghijklmnopqrstuvwxyzöäå-'

word_vec_length = 12 # Length of the input vector
char_vec_length = len(accepted_chars) # Length of the character vector
output_labels = 2 # Number of output labels

# Define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(accepted_chars))
int_to_char = dict((i, c) for i, c in enumerate(accepted_chars))

# Removes all non accepted characters
def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

# Returns a list of n lists with n = word_vec_length
def name_encoding(name):

    # Encode input data to int, e.g. a->1, z->26
    integer_encoded = [char_to_int[char] for i, char in enumerate(name) if i < word_vec_length]
    
    # Start one-hot-encoding
    onehot_encoded = list()
    
    for value in integer_encoded:
        # create a list of n zeros, where n is equal to the number of accepted characters
        letter = [0 for _ in range(char_vec_length)]
        letter[value] = 1
        onehot_encoded.append(letter)
        
    # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
    for _ in range(word_vec_length - len(name)):
        onehot_encoded.append([0 for _ in range(char_vec_length)])
        
    return onehot_encoded

# Encode the output labels
def label_encoding(gender_series):
    labels = np.empty((0, 2))
    for i in gender_series:
        if i == 1:
            labels = np.append(labels, [[1,0]], axis=0)
        else:
            labels = np.append(labels, [[0,1]], axis=0)
    return labels

def predict_unsure(row, col_name, model):
    if row[f'{col_name}_is_male'] == 0:
        name = row[col_name].split(' ')[0]
        encoded_name = np.asarray([name_encoding(normalize(name))])
        prediction = model.predict(encoded_name)[0]
        return 1 if prediction[0] > prediction[1] else -1
    
    else:
        return row[f'{col_name}_is_male']

def predict_irecommend(df, col_name = 'names'):
    
    neural_network = keras.models.load_model("gender_identifier")
    df[f'{col_name}_predicted_is_male'] = df.apply(predict_unsure, axis = 1, args = [col_name, neural_network])
    return df


def irecommend_df(csv_file_path : str, col_names : list , sensitive_columns : list = None, save_filepath = './bengts_data.csv'):
    
    df = pd.read_csv(csv_file_path)

    for col_name in col_names:
        df[f'{col_name}_is_male'] = df.apply(is_male, axis = 1, args = [col_name])

        if len(df[df[f'{col_name}_is_male'] == 0].index) > 0:
            df = predict_irecommend(df, col_name)

    # For Debugging:
    # Use df.info() in order to understand what the column names are 
    # df.info()
    
    if sensitive_columns:
        try:
            df = df.drop(columns = sensitive_columns)
        except:
            pass
    
    df.to_csv(save_filepath)
    
    return df

# Testing
# test_df = pd.DataFrame(boy_names, columns = ['names'])
# additional_names = pd.DataFrame(['fabian', 'jakob', 'marie', 'faraan', 'emily', 'linus', 'abdul', 'georgina', 'thomas pickett', 
# 'matthew', 'jamie', 'rasmus', 'freddy', 'frank', 'charlie', 'charli', 'charles', 'bengt',
# 'amelia', 'george', 'kurt', 'ingvar', 'ann', 'elisabeth'], columns = ['names'])
# test_df = test_df.append(additional_names)
# test_df.to_csv('./test.csv')
# asdf = irecommend_df('./test.csv', ['names'])
# asdf.to_csv('./test_answers.csv')
# print(asdf.tail())

# For fabian

# To run the program, first all that is needed to do is run the function irecommend_df
# The arguments that need to be passed are 
# 1. the filepath to the csv, 
# 2. the name of the column in which the first names are held. If these are first-name last-name, that is fine. If you do not know these column names, find the debug code and un-comment it to find out
# 3. sensitive columns is the names of the columns which are sensitive. This can be passed as a list, e.g ['email', 'phone', 'customerid', ...]
# 4. save_filepath is a filepath in which you want the file saved

# E.g
irecommend_df('./irecommend_sql.csv', ['givenname', 'referrer name'], ['email', 'phone', 'customerid'])