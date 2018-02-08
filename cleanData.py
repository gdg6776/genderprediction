"""
Program: cleanData.py
Author 1: Akshay Renavikar
Author 2: Gaurav Gawade
Description: This program performs the task of cleaning the data using the nltk library and generates the data
            file named clean.csv.
"""
import pandas as pd
import matplotlib.colors as colors
import re
from nltk.corpus import stopwords
import html
from nltk.stem import PorterStemmer
import numpy as np


# function to read file into pandas dataframe
def read(filename):
    """
    This function reads the file using pandas library and creates dataframe using columns specified in the
    fields list. Also, a call is made to the cleandata function.
    :param filename: Name of the file to be read.
    :return: None.
    """
    #read file with specified columns
    fields = ['description','link_color','name','sidebar_color','text','tweet_count','gender']
    dataframe = pd.read_csv(filename, encoding = "latin1", usecols=fields)
    cleandata(dataframe)


# function to perform data cleaning and preparation
def cleandata(dataframe):
    """
    This function performs the task of cleaning and preparation of data which includes deleting the columns
    which are not important while performing analysis.
    :param dataframe: The dataframe which contains the data after reading the CSV file in read function.
    :return: None.
    """
    # cleaning on target attribute
    dataframe = dataframe[dataframe.gender != 'brand']  # remove brand entries
    dataframe = dataframe[dataframe.gender != 'unknown']  # remove unknown entries
    dataframe = dataframe[pd.notnull(dataframe['gender'])]  # remove NA values

    # clean the text of three columns
    dataframe['Tweets'] = [strcleaning(s) for s in dataframe['text']]
    dataframe['Description'] = [strcleaning(s) for s in dataframe['description']]
    dataframe['Name'] = [strcleaning(s) for s in dataframe['name']]

    #normalize tweet_count column
    dataframe['tweetcount'] = (dataframe['tweet_count'] - dataframe['tweet_count'].min()) / (dataframe['tweet_count'].max() - dataframe['tweet_count'].min() )

    # remove old columns as these are replaced with new columns
    del dataframe['text']
    del dataframe['name']
    del dataframe['description']
    del dataframe['tweet_count']

    #cleaning the color columns to convert hex values to RGB
    dataframe = convertColor(dataframe)

    #arrange the columns for analysis
    dataframe = rearrange_columns(dataframe)
    #print(dataframe)
    dataframe.to_csv('clean.csv') # write dataframe to csv


# function to clean strings in the data
def strcleaning(s):
    """
    This functions is used to clean the strings which contains special characters and this is done
    using the PorterStemmer function.
    :param s: The text obtained from user's profile, tweet or description.
    :return: returns the cleaned string.
    """

    cachedStopWords = stopwords.words("english")
    st = PorterStemmer()
    s = str(s) #convert to string
    s = s.strip() # remove whitespaces
    s= s.encode('ascii', errors='ignore').decode('utf-8')#remove non-ascii characters like accented characters
    s = html.unescape(s) #remove html characters
    s = s.lower() #convert string to lowercase
    if s == "nan": #missing values handled by using a dummy string
        s = "dummy"
    s = s.replace("'","") #remove apostrophes
    s = s.replace(".","") #remove full stops
    s = s.replace(",", "") # remove commas
    s = s.replace("https", "") #remove links beginning with https
    s = re.sub("\d+", "", s)  # remove all numbers in the text
    # remove all other punctuations and symbols from string
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("[\w*"," ")
    s = ' '.join([word for word in s.split() if word not in cachedStopWords])# remove stopwords
    s = st.stem(s) #perform stemming
    return s


def convertColor(dataframe):
    """
    This function is used to convert the Hexadecimal values to colors.
    :param dataframe: The dataframe which contains the data after reading the CSV file in read function.
    :return: Dataframe obtained after converting color using convertColorHelper function.
    """
    names = ['link_color', 'sidebar_color']
    return convertColorHelper(dataframe, names)


def convertColorHelper(dataframe, color_list):
    """
    This is the helper function for convertColor wherein the padding is done to values whose length is less 
    than 6. After this the tuple is obtained which contains the values for Red, Green and Blue values of which the 
    index of the highest value is added to the list which is then added into the dataframe.
    :param dataframe: The dataframe which contains the data after reading the CSV file in read function.
    :param color_list: The names of the columns on which operation is performed.
    :return: Updated dataframe containg the values of colors.
    """
    tuple = ()
    hex_color = "#000000" #default black color used for missing values
    zero = "0"
    list = []
    for names in color_list:
        for data in dataframe[names]:
            if len(data) < 6: #if string is not in proper form
                number = 6 - len(data)
                new_data = zero.zfill(number) + data #append zeros at start
                tuple = colors.hex2color('#' + new_data) #convert to RGB
                list.append(color(tuple))
            elif len(data) == 6: #no cleaning needed
                tuple = colors.hex2color('#' + data)
                list.append(color(tuple))
            else: #use default color
                tuple = colors.hex2color(hex_color)
                list.append(color(tuple))
        m = np.asarray(list)
        dataframe[names] = m #if R value is maximum store color as Red, if Green is maximum store color as Green, similarly for blue
        list = []
    return dataframe


def color(tuple): #return index of max value of RGB tuple
    """
    This function is used to assign values to the indexes obtained from convertColorHelper function wherein,
    0 is for Red, 1 is for Green and 2 is for Blue.
    :param tuple: The tuple containing the values of Red, Green and Blue.
    :return: Name of the color.
    """
    color = tuple.index(max(tuple))
    if color == 0:
        return "Red"
    elif color == 1:
        return "Green"
    elif color == 2:
        return "Blue"


def rearrange_columns(dataframe):
    """
    This function is basically used to rearrange the columns in the datagframe.
    :param dataframe: The dataframe which contains the data after reading the CSV file in read function.
    :return: updated dataframe.
    """
    dataframe = dataframe[['Name', 'Description',  'Tweets', 'link_color', 'sidebar_color',
                 'tweetcount', 'gender']]
    return dataframe


def main():
    """
    The main function which makes a call to read fucntion.
    :return: None.
    """
    filename = "data.csv"
    read(filename)

if __name__ == '__main__':
    main()