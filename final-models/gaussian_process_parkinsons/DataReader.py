#############################################################################
# CSV_DataReader.py
#
# This program is the data reading code of the Naive Bayes classifier from week 1.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
#
# Version: 1.0, Date: 20 September 2024 code decoupled from NBClassifier
# Version: 1.1, Date: 23 October 2024 Support for standardising continous data
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import numpy as np
from sklearn.preprocessing import StandardScaler


class CSV_DataReader:
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    scaler = None

    def __init__(self, file_name, stardardise_data, pretrained_scaler):
        if file_name is None:
            return
        else:
            if pretrained_scaler is not None: 
                self.scaler = pretrained_scaler
            self.read_data(file_name, stardardise_data)

    def read_data(self, data_file, stardardise_data):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = []
                else:
                    values = line.split(',')

                    if len(self.rv_all_values) == 0:
                        continuous_inputs = self.check_datatype(values)
                        print("continuous_inputs="+str(continuous_inputs))

                    if continuous_inputs is True:
                        values = [float(value) for value in values]

                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1

                if self.num_data_instances>=2000: break

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        if stardardise_data is True and continuous_inputs is True:
            self.rv_all_values = self.standardise_data(self.rv_all_values, data_file, )

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % ("omitted due to size"))#self.rv_key_values))
        print("VARIABLE VALUES=%s" % ("omitted due to size"))#self.rv_all_values))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)

    def standardise_data(self, data, datafile):
        print("NORMALISING inputs of datafile=%s..." % (datafile))
        data = np.asarray(data)
        # initialises a vector of feature values, without the target random variable
        """
        X_normalised = np.zeros(X.shape)
        for i in range(0, len(X[0])):
            if i == len(X[0])-1:
                X_normalised[:,i] = X[:,i]#.astype(int)
            else:
                #X_column_i = X[:,i]
                #_mean = np.mean(X_column_i)
                #_std = np.std(X_column_i)
                #X_normalised[:,i] = (X_column_i-_mean)/(_std)
                X_normalised[:,i] = X[:,i]
        """
        # normalisation: substracts mean and divides by standard deviation
        X = data[:, :-1]  # All columns except the last
        Y = data[:, -1]   # Only the last column
        if self.scaler is None: # scaler on training data
            self.scaler = StandardScaler()
            X_normalised = self.scaler.fit_transform(X)
        else: # test data uses pretrained scaler
            X_normalised = self.scaler.transform(X)
        data_normalised = np.column_stack((X_normalised, Y))
        print("data_normalised=",data_normalised)
        return data_normalised

    def check_datatype(self, values):
        for feature_value in values:
            if feature_value[0].isalpha():
                return False # discrete data (due to values being alphabetic characters)
            elif len(feature_value.split('.')) > 1 or len(feature_value) > 1:
                return True # continuous data (due to decimals or values above digits)
        return False # discrete data (due to not finding decimals and only digits)
