"""
Program: old.py
Author 1: Akshay Renavikar
Author 2: Gaurav Gawade
Description: This program performs the task of classifying the data using Naive Bayes Algorithm
            along with Visualization of results and accuracy for both classification Algorithm.
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# function to read file into pandas dataframe
def read(filename):
    """
    This function is used to read file from system and this is done using Pandas library which generates 
    the dataframe. Also, a call is made to transformText and findWords function.
    :param filename: Name of the file to be processed.
    :return: None.
    """

    #read file with specified columns
    dataframe = pd.read_csv(filename, encoding = "latin1")

    # create a new column as a concatenation of all columns for bag of words representation
    dataframe['allFeatures'] = dataframe['text'].str.cat(dataframe['description'], sep=' ').str.cat(dataframe['sidebar_color'],sep=' ')
    dataframe['allFeatures'] = dataframe['allFeatures'].str.cat(dataframe['link_color'], sep =' ').str.cat(dataframe['name'],sep=' ')
    transformText(dataframe)


#function to transform text to bag of words
def transformText(dataframe):
    """
    This functions generates bag of words which are transformed from text using the Tfid and then 
    the data is split into training and test data which are provided as a parameter to the Classify function.
    :param dataframe: The datafrane containing the cleaned data.
    :return: None.
    """

    # encoding text data to form structure like bag of words
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(dataframe['allFeatures'].values.astype('U'))
    #encode the target label
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataframe['gender'].values.astype('U'))

    #split data into training and testing
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
    classify(x_train, x_test, y_train, y_test)


# function to use Naive Bayes algorithm to classify the text
def classify(x_train, x_test, y_train, y_test):
    """
    This function performs the task of classification using the Naive Bayes approach along with displaying the results
    using graphs. The major componentes calculated in this fucntion are Accuracy, Precision, Recall and F1 Score.
    :param x_train: Features of Training data.
    :param x_test: Features of Trainging data.
    :param y_train: Target varaible of training data.
    :param y_test: Target of test data.
    :return: None.
    """
    nb = MultinomialNB()
    nb.fit(x_train, y_train)  # training model
    #print(nb.score(x_test, y_test)) # score for classification


def main():
    """
    The main function which makes a call to read fucntion.
    :return: None.
    """
    filename = "data.csv"
    read(filename)


if __name__ == '__main__':
    main()