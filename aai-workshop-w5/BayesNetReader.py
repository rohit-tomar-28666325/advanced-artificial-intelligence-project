#############################################################################
# BayesNetReader.py
#
# Reads a configuration file containing the specification of a Bayes net.
# It generates a dictionary of key-value pairs containing information
# describing the random variables, structure, and conditional probabilities.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
## 
# Keys expected: name, random_variables, structure, and CPTs.
# Separators: COLON(:) for key-values, EQUALS(=) for table_entry-probabilities
# The following is a snippet of configuration file config_alarm.txt
# --------------------------------------------------------
# name:Alarm
# 
# random_variables:Burglary(B);Earthquake(E);Alarm(A);JohnCalls(J);MaryCalls(M)
# 
# structure:P(B);P(E);P(A|B,E);P(J|A);P(M|A)
# 
# CPT(B):
# true=0.001;false=0.999

# CPT(E):
# true=0.002;false=0.998
# 
#  ...
# 
# CPT(M|A):
# true|true=0.70;
# true|false=0.01;
# false|true=0.30;
# false|false=0.99
# --------------------------------------------------------
# The file above replaces CPTs by regression_models in the case of Bayes nets
# with continuous data, where instead of CPTs regression models are used.
#
# Version: 1.0
# Date: 06 October 2022 first version
# Date: 25 October 2023 extended for Bayes nets with KernelRidge regression
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import pickle


class BayesNetReader:
    bn = {}

    def __init__(self, file_name):
        self.read_data(file_name)
        self.tokenise_data()
        self.load_regression_models()

    # starts loading a configuration file into dictionary 'bn', by
    # splitting strings with character ':' and storing keys and values 
    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))

        with open(data_file) as cfg_file:
            key = None
            value = None
            for line in cfg_file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(":")
                if len(tokens) == 2:
                    if value is not None:
                        self.bn[key] = value
                        value = None

                    key = tokens[0]
                    value = tokens[1]
                else:
                    value += tokens[0]

        self.bn[key] = value
        self.bn["random_variables_raw"] = self.bn["random_variables"]
        print("RAW key-values="+str(self.bn))

    # continues loading a configuration file into dictionary 'bn', by
    # separating key-value pairs as follows:
    # (a) random_variables are stored as list in self.bn['random_variables']
    # (b) CPTs are stored as an inner dictionary in self.bn['CPT']
	# (c) all others are stored as key-value pairs in self.bn[key]
    def tokenise_data(self):
        print("TOKENISING data...")
        rv_key_values = {}

        for key, values in self.bn.items():

            if key == "random_variables":
                var_set = []
                for value in values.split(";"):
                    if value.find("(") and value.find(")"):
                        value = value.replace('(', ' ')
                        value = value.replace(')', ' ')
                        parts = value.split(' ')
                        var_set.append(parts[1].strip())
                    else:
                        var_set.append(value)
                self.bn[key] = var_set

            elif key.startswith("CPT"):
                # store Conditional Probability Tables (CPTs) as dictionaries
                cpt = {}
                sum = 0
                for value in values.split(";"):
                    pair = value.split("=")
                    cpt[pair[0]] = float(pair[1])
                    sum += float(pair[1])
                print("key=%s cpt=%s sum=%s" % (key, cpt, sum))
                self.bn[key] = cpt

                # store unique values for each random variable
                if key.find("|") > 0:
                    rand_var = key[4:].split("|")[0]
                else:
                    rand_var = key[4:].split(")")[0]
                unique_values = list(cpt.keys())
                rv_key_values[rand_var] = unique_values

            else:
                values = values.split(";")
                if len(values) > 1:
                    self.bn[key] = values

        self.bn['rv_key_values'] = rv_key_values
        print("TOKENISED key-values="+str(self.bn))

    # puts the following key-value pairs in 'bn' as follows:
    # means of each random variable in regression_models['means']
    # standard deviations of random variables in regression_models['stdevs']
    # regression models of random variables in regression_models['coefficients']
    def load_regression_models(self):
        # check whether the regression_models exist (defined in the config file)
        is_regression_models_available = False
        for key, value in self.bn.items():
            if key == "regression_models":
                is_regression_models_available = True
				
        # loads the regression_models as per the .pkl file in the config file
        if  is_regression_models_available:
            try:
                configfile_name = self.bn["regression_models"]
                print("\nLOADING %s ..." % (configfile_name))
                models_file = open(configfile_name, 'rb')
                regression_models = pickle.load(models_file)
                self.bn["means"] = regression_models["means"]
                self.bn["stdevs"] = regression_models["stdevs"]
                self.bn["regressors"] = regression_models["regressors"]
				
                models_file.close()
                print("Regression models loaded!")

            except Exception:
                print("Couldn't find file %s" % (configfile_name))
                pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: BayesNetReader.py [your_config_file.txt]")
    else:
        file_name = sys.argv[1]
        BayesNetReader(file_name)
