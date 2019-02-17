'''
classifier to classify if a given patient has
heart disease using the features like age, sex, etc.

Date: Feb 15, 2019
Author: Yao Li
'''
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

class classifier:

    #init with 13 input parameters
    def __init__(self, input_data, model_path, data_path):
        self.input_data = input_data
        self.model = self.load_model(model_path, data_path)
        #self.data
        #self.label

    #load the model
    def load_model(self, model_path, data_path):

        model = pickle.load(open(model_path, 'rb'))
        data = pd.read_csv(data_path)
        #self.label = data['target'].values
        #self.data = data.drop(['target'], axis=1)

        return model

    #diagnose a new patient
    def predit(self):
        #self.input_data[0] = (self.input_data[0] - np.min(self.data)) / (np.max(self.data) - np.min(self.data)).values
        res = self.model.predict(self.input_data)
        print(res[0])

        return


def main():
    #test_case = np.asarray([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    test_case = np.asarray([[58, 1, 0, 100, 234, 0, 1, 156, 0, 0.1, 2, 1, 3]])
    diag = classifier(test_case, '../model/model.sav', '../data/heart.csv')
    diag.predit()


if __name__ == '__main__':
    main()
