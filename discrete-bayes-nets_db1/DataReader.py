#############################################################################
# CSV_DataReader.py
#
# This program is the data reading code of the Naive Bayes classifier from week 1.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
#
# Version: 1.0, Date: 20 September 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################


class CSV_DataReader:
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0

    def __init__(self, file_name):
        if file_name is None:
            return
        else:
            self.read_data(file_name)

    def read_data(self, data_file):
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
                    self.rv_all_values.append(values)
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1]

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES=%s" % (self.rv_all_values))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i]
            key_values = self.rv_key_values[variable]
            value_in_focus = values[i]
            if value_in_focus not in key_values:
                self.rv_key_values[variable].append(value_in_focus)
