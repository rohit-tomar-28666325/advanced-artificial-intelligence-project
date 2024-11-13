#############################################################################
# ModelEvaluator.py
#
# Implements the following performance metrics and scoring functions:
# Balanced Accuracy, F1 Score, Area Under Curve (AUC), 
# Brier Score, Kulback-Leibler Divergence (KLL), training/test times.
# Log Likelihood (LL), Bayesian Information Criterion (BIC).
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
# Version: 1.5, Date: 23 October 2024 support for gpytorch Gaussian Proceess regression
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import os
import sys
import math
import time
import random
import numpy as np
#import torch
#import gpytorch
#from GPRegressionModel import GPRegressionModel
from sklearn import metrics

import BayesNetUtil as bnu
from DataReader import CSV_DataReader
from BayesNetInference import BayesNetInference
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import label_binarize


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
        if self.bn.get("scaler", None) is not None:
            print("READING test data using pretrained scaler...")
            self.csv = CSV_DataReader(datafile_test, True, self.bn["scaler"])
        else: 
            self.csv = CSV_DataReader(datafile_test, False, None)

        # generates performance results from the predictions above  
        self.inference_time = time.time()
        #self.initialise_models_and_likelihoods()
        true, pred, prob = self.get_true_and_predicted_targets()
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(true, pred, prob)

    """
    def initialise_models_and_likelihoods(self):
        self.bn["models_and_likelihoods"] = {}
        self.bn["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for V in self.bn["random_variables"]:
            if self.bn["regressors"].get(V) is not None:
                regressor = self.bn["regressors"][V]
                try:
                    if 'model' in regressor and 'likelihood' in regressor:
                        inputs = regressor['inputs']
                        outputs = regressor['outputs']
                        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.bn["device"])
                        model = GPRegressionModel(inputs, outputs, likelihood).to(self.bn["device"])
                        model.load_state_dict(regressor['model'])
                        likelihood.load_state_dict(regressor['likelihood'])
                        model.eval()
                        likelihood.eval()
                        self.bn["models_and_likelihoods"][V] = {"model": model, "likelihood": likelihood}
                        print("Done initialising model and likelihood of variable %s" % (V))
                except Exception:
                    print("Not attempting to load a gpytorch model!")
                    pass
    """

    def get_true_and_predicted_targets(self):
        print("\nPERFORMING probabilistic inference on test data...")
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtains vectors of categorical and probabilistic predictions
        # but only for binary classification -- needs extension for multiclass classification
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
            prob_dist = self.get_predictions_from_BayesNet(data_point)

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
    def get_predictions_from_BayesNet(self, data_point):
        # forms a probabilistic query based on the predictor variable,
        # the evidence (non-predictor variables), and the values of
        # the current data point (test instance) given as argument
        evidence = ""
        for var_index in range(0, len(self.csv.rand_vars)-1):
            evidence += "," if len(evidence)>0 else ""
            evidence += self.csv.rand_vars[var_index]+'='+str(data_point[var_index])
        prob_query = "P(%s|%s)" % (self.csv.predictor_variable, evidence)
        self.query = bnu.tokenise_query(prob_query, False)
        if self.verbose: print("self.query=",self.query)

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
        
        # def multiclass_brier_score(Y_true, Y_prob):
        #         # Ensure Y_true is a NumPy array
        #     Y_true = np.array(Y_true)
            
        #     # Convert Y_prob to a 2D NumPy array if it's not already
        #     Y_prob = np.array(Y_prob)
            
        #     n_classes = len(np.unique(Y_true))  # Determine the number of classes
        #     Y_prob_2d = np.zeros((len(Y_true), n_classes))  # Initialize a 2D array for probabilities

        #     # Assuming you have some logic to fill Y_prob_2d
        #     # Here we will assume that Y_prob is the predicted probability of the first class
        #     for i in range(len(Y_true)):
        #         Y_prob_2d[i, Y_true[i]] = Y_prob[i]  # Assign the predicted probability to the correct class index

        #     brier_scores = []
        #     for class_label in range(n_classes):
        #         binary_true = (Y_true == class_label).astype(int)  # Ensure this is a NumPy array
        #         binary_prob = Y_prob_2d[:, class_label]  # Now this will work
        #         score = brier_score_loss(binary_true, binary_prob)
        #         brier_scores.append(score)
            
        #     return np.mean(brier_scores)
        
        # def auc_multiclass( Y_true, Y_prob):
        #     Y_prob = np.array(Y_prob).reshape(-1, 1)  # Reshape into a 2D array
        #     Y_prob = np.hstack((1 - Y_prob, Y_prob * 0.5, Y_prob))

        #     # Binarize the true labels
        #     Y_true_binarized = label_binarize(Y_true, classes=[0, 1, 2])

        #     # Calculate AUC for each class and average macro
        #     return roc_auc_score(Y_true_binarized, Y_prob, average="macro", multi_class="ovr")
        
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        # print("Y_true="+str(Y_true))
        # print("Y_pred="+str(Y_pred))
        # print("Y_prob="+str(Y_prob))
        
        print(len(Y_true),len(Y_pred),len(Y_prob))
        # return
        accuracy = metrics.accuracy_score(Y_true,Y_pred)
        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        # f1 = metrics.f1_score(Y_true, Y_pred, average='weighted')
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # auc = self.auc_multiclass(Y_true, Y_prob)
        
        
        # print(Y_true)
        # print(Y_pred)
        # print(Y_prob)

       

        
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        # brier = multiclass_brier_score(Y_true, Y_prob)  # Use the custom function for Brier score
        kl_div = np.sum(P*np.log(P/Q))

        print("\nCOMPUTING performance on test data...")

        print("Accuracy="+str(accuracy))
        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))		
        print("Training Time=this number should come from the CPT_Generator!")
        # print("Inference Time="+str(self.inference_time)+" secs.")
        return {
            "accuracy": accuracy,
            "bal_acc": bal_acc,
            "f1_score": f1,
            "auc": auc,
            # "brier": brier,
            "kl_div": kl_div,
            # "inference_time": self.inference_time
        }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: ModelEvaluator.py [config_file.txt] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_test = sys.argv[2]
        ModelEvaluator(configfile, datafile_test)
