#############################################################################
# ModelEvaluator.py
#
# Implements the following scoring functions and performance metrics:
# Log Likelihood (LL), Bayesian Information Criterion (BIC).
# Balanced Accuracy, F1 Score, Area Under Curve (AUC), 
# Brier Score, Kulback-Leibler Divergence (KLL), training/test times.
#
# IMPORTANT: This program currently makes use of two instantiations of
# NB_Classifier: one for training and one for testing. If you want this
# program to work for any arbitrary Bayes Net, the constructor (__init__) 
# needs to be updated to support a trainer (via CPT_Generator) and a
# tester (e.g., via BayesNetExactInference) -- instead of Naive Bayes models.
#
# This implementation also assumes that normalised probability distributions
# of predictions are stored in an array called "NB_Classifier.predictions".
# Performance metrics need such information to do the required calculations.
#
# This program has been tested for Binary classifiers. Minor extensions are
# needed should you wish this program to work for non-binary classifiers.
#
# Version: 1.0, Date: 03 October 2022, basic functionality
# Version: 1.1, Date: 15 October 2022, extended with performance metrics
# Version: 1.2, Date: 18 October 2022, extended with LL and BIC functions (removed)
# Version: 1.3, Date: 21 October 2023, refactored for increased reusability 
# Version: 1.4, Date: 22 September 2024, Naive Bayes removed to focus on Bayes nets
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import random
import numpy as np
import os.path
from sklearn import metrics

import BayesNetUtil as bnu
from DataReader import CSV_DataReader
from BayesNetInference import BayesNetInference


class ModelEvaluator(BayesNetInference):
    verbose = False 
    inference_time = None

    def __init__(self, configfile_name, datafile_test):
        if os.path.isfile(configfile_name):
            # loads Bayesian network stored in configfile_name, where
            # the None arguments prevent any inference at this time
            super().__init__(None, configfile_name, None, None)
            self.inference_time = time.time()

        # reads test data using code from DataReader
        self.csv = CSV_DataReader(datafile_test)

        # generates performance results from the predictions above  
        self.inference_time = time.time()
        true, pred, prob = self.get_true_and_predicted_targets()
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(true, pred, prob)

    def get_true_and_predicted_targets(self):
        print("\nCARRYING-OUT probabilistic inference on test data...")
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtains vectors of categorical and probabilistic predictions
        for i in range(0, len(self.csv.rv_all_values)):
            data_point = self.csv.rv_all_values[i]
            target_value = data_point[len(self.csv.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)
            elif target_value == 1: Y_true.append(1)
            elif target_value == 0: Y_true.append(0)

            # obtains a probability distribution of predictions as a dictionary 
            # either from a Bayesian Network or from a Naive Bayes classifier.
            # example prob_dist={'1': 0.9532340821183165, '0': 0.04676591788168346}
            prob_dist = self.get_predictions_from_BayesNet(data_point, self.csv)

            # retrieves the probability of the target_value and adds it to
            # the vector of probabilistic predictions referred to as 'Y_prob'
            try:
                predicted_output = prob_dist[target_value]
            except Exception:
                predicted_output = prob_dist[float(target_value)]
            if target_value in ['no', '0', 0]:
                predicted_output = 1-predicted_output
            Y_prob.append(predicted_output)

            # retrieves the label with the highest probability, which is
            # added to the vector of hard (non-probabilistic) predictions Y_pred
            # this is only for binary classification -- needs extension for multiclass classification
            best_key = max(prob_dist, key=prob_dist.get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)
            elif best_key == 1: Y_pred.append(1)
            elif best_key == 0: Y_pred.append(0)

        # verifies that probabilities are not NaN (not a number) values -- 
        # in which case are replaced by 0 probabilities
        for i in range(0, len(Y_prob)):
            if np.isnan(Y_prob[i]):
                Y_prob[i] = 0

        return Y_true, Y_pred, Y_prob

    # returns a probability distribution using Inference By Enumeration
    def get_predictions_from_BayesNet(self, data_point, nbc):
        # forms a probabilistic query based on the predictor variable,
        # the evidence (non-predictor variables), and the values of
        # the current data point (test instance) given as argument
        evidence = ""
        for var_index in range(0, len(self.csv.rand_vars)-1):
            evidence += "," if len(evidence)>0 else ""
            evidence += self.csv.rand_vars[var_index]+'='+str(data_point[var_index])
        prob_query = "P(%s|%s)" % (self.csv.predictor_variable, evidence)
        self.query = bnu.tokenise_query(prob_query, False)

        # sends query to BayesNetInference and get probability distribution
        self.prob_dist = self.enumeration_ask()
        normalised_dist = bnu.normalise(self.prob_dist)
        if self.verbose: print("%s=%s" % (prob_query, normalised_dist))

        return normalised_dist

    # prints model performance according to the following metrics:
    # balanced accuracy, F1 score, AUC, Brier score, KL divergence,
    # and training and test times. But note that training time is
    # dependent on model training externally to this program, which
    # is the case of Bayes nets trained via CPT_Generator.py	
    def compute_performance(self, Y_true, Y_pred, Y_prob):
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        #print("Y_true="+str(Y_true))
        #print("Y_pred="+str(Y_pred))
        #print("Y_prob="+str(Y_prob))

        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))

        print("\nCOMPUTING performance on test data...")

        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))		
        print("Training Time=this number should come from the CPT_Generator!")
        print("Inference Time="+str(self.inference_time)+" secs.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        ModelEvaluator(configfile, datafile_test)
