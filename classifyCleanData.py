"""
Program: classifyCleanData.py
Author 1: Akshay Renavikar
Author 2: Gaurav Gawade
Description: This program performs the task of classifying the data using Naive Bayes and Random Forest Algorithm
            along with Visualization of results and accuracy for both classification Algorithm.
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


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
    dataframe['allFeatures'] = dataframe['Tweets'].str.cat(dataframe['Description'], sep=' ').str.cat(dataframe['sidebar_color'],sep=' ')
    dataframe['allFeatures'] = dataframe['allFeatures'].str.cat(dataframe['link_color'], sep =' ')#.str.cat(dataframe['Name'],sep=' ')
    transformText(dataframe)
    findWords(dataframe)


#function to find the words and color for each gender
def findWords(dataframe):
    """
    This function is used to find the words from the pandas dataframe such as male and female and the count
    of the number of words is displayed as the output.
    :param dataframe: The datafrane containing the cleaned data.
    :return: None.
    """
    Male = dataframe[dataframe['gender'] == 'male']
    Female = dataframe[dataframe['gender'] == 'female']
    Male_Words = pd.Series(' '.join(Male['Tweets'].astype(str)).lower().split(" ")).value_counts()[:40]
    Female_Words = pd.Series(' '.join(Female['Tweets'].astype(str)).lower().split(" ")).value_counts()[:40]
    Male_Colors = pd.Series(' '.join(Male['sidebar_color'].astype(str)).lower().split(" ")).value_counts()[:40]
    Female_Colors = pd.Series(' '.join(Female['sidebar_color'].astype(str)).lower().split(" ")).value_counts()[:40]
    print("Male Words: ")
    print(Male_Words)
    print("\n")
    print("Female Words: ")
    print(Female_Words)
    print("\n")
    print("Male Colors: ")
    print(Male_Colors)
    print("\n")
    print("Female Colors: ")
    print(Female_Colors)


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
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.5)
    classifyNB(x_train, x_test, y_train, y_test)
    classifyRF(x_train, x_test, y_train, y_test)


# function to use Naive Bayes algorithm to classify the text
def classifyNB(x_train, x_test, y_train, y_test):
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
    output = nb.predict(x_test)  # running model on test data

    #calculating metrics on the results
    acc = accuracy_score(y_test, output)  # calculate accuracy of model
    print("Accuracy of Naive Bayes Model is:", acc)
    np.set_printoptions(suppress=True)
    pre = precision_score(y_test, output)  # calculate precision score of model
    print("Precision of Naive Bayes Model is:", pre)
    rec = recall_score(y_test, output)  # calculate recall score of model
    print("Recall of Naive Bayes Model Model is:", rec)
    f1 = f1_score(y_test, output)  # calculate f1 score of model
    print("F1 score of Naive Bayes Model is:", f1)

    #plotting the ROC curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, output)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC Curve for Naive Bayes')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("\n")


# function to use Random forest algorithm to classify the text
def classifyRF(x_train, x_test, y_train, y_test):
    """
    This function performs the task of classification using the Random Forest approach along with displaying the results
    using graphs. The major componentes calculated in this fucntion are Accuracy, Precision, Recall and F1 Score.
    :param x_train: Features of Training data.
    :param x_test: Features of Trainging data.
    :param y_train: Target varaible of training data.
    :param y_test: Target of test data.
    :return: None.
    """

    rf = RandomForestClassifier(n_estimators=15, criterion="gini")
    # change the above parameters to obtain different models
    rf.fit(x_train, y_train)  # training model
    output = rf.predict(x_test)  # running model on test data
    acc = accuracy_score(y_test, output)  # calculate accuracy of model
    print("Accuracy of Random Forest Classifier Model is:", acc)
    np.set_printoptions(suppress=True)  # code to disable scientific notation from output
    pre = precision_score(y_test, output)  # calculate precision score of model
    print("Precision of Random Forest Classifier Model is:", pre)
    rec = recall_score(y_test, output)  # calculate recall score of model
    print("Recall of Random Forest Classifier Model is:", rec)
    f1 = f1_score(y_test, output)  # calculate f1 score of model
    print("F1 score of Random Forest Classifier Model is:", f1)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, output)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC Curve for Random Forest Classifier')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("\n")


def main():
    """
    The main function which makes a call to read fucntion.
    :return: None.
    """
    filename = "clean.csv"
    read(filename)


if __name__ == '__main__':
    main()