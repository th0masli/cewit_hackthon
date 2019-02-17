'''
train heart.csv dataset to get a heart disease classification model
save the model to the specified path

Date: Feb 15, 2019
Author: Yao Li
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class model:

    #init
    def __init__(self):
        #init training data
        self.train_x = 0
        self.train_y = 0
        #init test data
        self.test_x = 0
        self.test_y = 0

    '''
    data loader
    load data file to
    train and test
    '''
    def load_data(self, data_path):
        data = pd.read_csv(data_path)
        #split the data and label
        y = data.target.values
        x = data.drop(['target'], axis=1)
        #normalize the data
        #x = (x - np.min(x)) / (np.max(x) - np.min(x)).values
        #split the data in to train and test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.2, random_state=0)

        #return void

    #clean data

    #train data
    def train(self):
        clf = LogisticRegression()
        clf.fit(self.train_x, self.train_y)
        if self.test(clf) > 80:
            self.save_model(clf)

        return

    #test data
    def test(self, clf):
        accuracy = clf.score(self.test_x, self.test_y)*100
        print("Accuracy: {:.2f}%".format(accuracy))

        return accuracy

    #save model to specified path
    def save_model(self, clf):
        clf_path = '../model/model.sav'
        pickle.dump(clf, open(clf_path, 'wb'))

        return


def main():
    lr_model = model()
    lr_model.load_data('../data/heart.csv')
    lr_model.train()


if __name__ == '__main__':
    main()
