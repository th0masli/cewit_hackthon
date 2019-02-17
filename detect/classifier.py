'''
classifier to classify if a given patient has
heart disease using the features like age, sex, etc.

Date: Feb 15, 2019
Author: Yao Li
'''
import numpy as np
import pickle

class classifier:

    #init with 13 input parameters
    def __init__(self, input_data, model_path):
        self.input_data = input_data
        self.model = self.load_model(model_path)

    #load the model
    def load_model(self, model_path):

        model = pickle.load(open(model_path, 'rb'))

        return model

    #diagnose a new patient
    def predict(self):
        res = self.model.predict(self.input_data)
        #print(res[0])

        return res[0]
