#############################################################################
# gpyroch_GPR.py
#
# Implements functionality for Gaussian Process classification via GP regression
#
# This program is a baseline classifier with class predictions derived from
# estimated mean vector and covariance matrices of a GPR according to
#     pdf_1 = self.get_gaussian_probability_density(1, pred_mean, pred_std)
#     pdf_0 = self.get_gaussian_probability_density(0, pred_mean, pred_std)
#     prob = pdf_1 / (pdf_1 + pdf_0)
#
# This program can run on CPU or GPU devices, as detected at runtime.
# Key software dependencies:
#     pip install torch --index-url https://download.pytorch.org/whl/cu118
#     pip install gpytorch
#
# Since the above assumes an existing installation of CUDA version 11.8, you
# need to verify your version of CUDA to be able to run the code on you GPU. 
# Ignore the CUDA part if you are instested in CPU execution.
#
# Version: 1.0, Date: 25 October 2024, functionality tested on multiple datasets
#               for binary classification -- and coupled with ModelEvaluator.py 
# Version: 1.1, Date: 26 October 2024, support for plotting data in 3D     
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import time
import torch
import gpytorch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics

import torch
import gpytorch
from gpytorch.kernels import Kernel




class CustomMaternKernel(Kernel):
    def __init__(self, nu=2.5, lengthscale=1.0, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        # Register lengthscale and variance as parameters
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.tensor(variance)))
        
    def forward(self, x1, x2, diag=False, **params):
        # Compute the Euclidean distance
        dists = torch.cdist(x1, x2) / self.raw_lengthscale
        
        if self.nu == 0.5:  # Matern 1/2 is equivalent to the absolute exponential kernel
            matern_term = torch.exp(-dists)
        elif self.nu == 1.5:
            sqrt3 = torch.sqrt(torch.tensor(3.0))
            matern_term = (1 + sqrt3 * dists) * torch.exp(-sqrt3 * dists)
        elif self.nu == 2.5:
            sqrt5 = torch.sqrt(torch.tensor(5.0))
            matern_term = (1 + sqrt5 * dists + (5.0 / 3.0) * dists ** 2) * torch.exp(-sqrt5 * dists)
        else:
            raise ValueError(f"Matern kernel with nu={self.nu} is not implemented.")
        
        # Apply variance scaling
        return self.raw_variance * matern_term



# Gaussian Process regressor using the Radial Basis Function kernel (others possible)
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel()
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5) # matern12
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) # matern32
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5) # matern52
        self.covar_module = CustomMaternKernel(nu=2.5)

        # self.covar_module = gpytorch.kernels.RQKernel(alpha=1.0)
        # self.covar_module = gpytorch.kernels.PolynomialKernel(2)
        # self.covar_module = gpytorch.kernels.PolynomialKernel(3)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Gaussian Process regressor applied to classification tasks
class GPR():
    MAX_TRAIN_DATA=2000
    STANDARDISE_DATA = True
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    filename = 'gp_model_and_likelihood.pkl'
    REDUCE_INPUTS_TO_2D = False
    device = None
    pca = None

    def __init__(self, datafile_train, datafile_test):
        self.perfromance = {}
        self.dataset_column = []
        self.training_time = None
        # Load training and test data from two separate CVS files
        X_train, Y_train = self.load_data(datafile_train, True) # False to use all training data
        X_test, Y_test = self.load_data(datafile_test, True) # False to use the entire test set
        X_train, X_test = self.get_standardised_data(X_train, X_test)
        X_train, X_test = self.get_data_with_PCA(X_train, X_test)
        print("%s training instances" % (len(X_train)))
        print("%s test instances" % (len(X_test)))
        # Convert the data to PyTorch tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        X_train = torch.Tensor(X_train).to(self.device)
        Y_train = torch.Tensor(Y_train.values).to(self.device)
        X_test = torch.Tensor(X_test)
        Y_test = torch.Tensor(Y_test.values)

		# train GP model via regression and evaluate it with test data
        model, likelihood, training_time = self.train_GPR(X_train, Y_train)
        #self.save_GPR(self.filename, model, likelihood)
        #model, likelihood = self.load_GPR(self.filename, X_train, Y_train)
        self.model, self.likelihood  = self.evaluate_GPR(X_test, Y_test, model, likelihood, training_time)
        self.plot_GPR(X_train, Y_train, X_test, Y_test, self.model, self.likelihood )
      
    def load_data(self, csv_file, useDataSampling_NotFullSet=False):
        print("LOADING and PROCESSING data...")
        df = pd.read_csv(csv_file, encoding='latin')
        self.dataset_column = df.columns
        X = df.iloc[:, :-1]  
        Y = df.iloc[:, -1]   
        if useDataSampling_NotFullSet and len(X)>self.MAX_TRAIN_DATA:
            random_indices = np.random.choice(X.shape[0], self.MAX_TRAIN_DATA, replace=False)
            X = X.iloc[random_indices]
            Y = Y.iloc[random_indices]
        return X, Y

    # equation of the Univariate Gaussian distribution, using mean and var
    def get_gaussian_density_from_var(self, x, mean, var):
        e_val = -np.power((x-mean), 2)/(2*var)
        return (1/(np.sqrt(2*np.pi*var))) * np.exp(e_val)

    # equation of the Univariate Gaussian distribution, using mean and std
    def get_gaussian_density_from_std(self, x, mean, stdev):
        e_val = -0.5*np.power((x-mean)/stdev, 2)
        probability = (1/(stdev*np.sqrt(2*np.pi))) * np.exp(e_val)
        return probability

    # standardize the features 
    def get_standardised_data(self, X_train, X_test):
        print("STANDARDISE_DATA=%s" % (self.STANDARDISE_DATA))
        if self.STANDARDISE_DATA:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
        return X_train, X_test

    # reduce the data, for visualisation purposes, using Principal Component Analysis (PCA) 
    def get_data_with_PCA(self, X_train, X_test):  
        if self.REDUCE_INPUTS_TO_2D is True:
            print("Applying PCA dimentionality reduction to the data...")
            self.pca = PCA(n_components=2)
            X_train_reduced = self.pca.fit_transform(X_train)
            X_test_reduced = self.pca.transform(X_test)
            return X_train_reduced, X_test_reduced
        else:
            return X_train, X_test

    # training procedure for the GPR 
    def train_GPR(self, X_train, y_train):
        print("\nTRAINING Gaussian Process model...")
        training_time = time.time()

        # Initialize the likelihood and model. Whilst the former defines the noise, 
        # the later is used to learn the mean vector and covariance matrix. 
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPRegressionModel(X_train, y_train, likelihood).to(self.device)

        # Set the model in training mode
        model.train()
        likelihood.train()

        # Optimize the model hyperparameters via the MLL loss function
        print("Loss function: ExactMarginalLogLikelihood")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        for i in range(self.NUM_EPOCHS):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Iteration {i+1}: Loss = {loss.item()}")

        self.training_time = time.time() - training_time
        return model, likelihood, training_time

    # omited code but left here for future possible uses when training and test
    # could be done at different times, e.g., train once and test multiple times
    """
    def save_GPR(self, filename, model, likelihood):
        with open(filename, 'wb') as models_file:
            regression_models = {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict()
            }
            pickle.dump(regression_models, models_file)
        print("GP model saved!")

    def load_GPR(self, filename, X_train, y_train):
        # initialise model and likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPRegressionModel(X_train, y_train, likelihood).to(self.device)

        # assign the learnt states, during training, to model and likelihood
        models_file = open(filename, 'rb')
        regression_models = pickle.load(models_file)
        model.load_state_dict(regression_models["model_state_dict"])
        likelihood.load_state_dict(regression_models["likelihood_state_dict"])
        models_file.close()
        return model, likelihood
    """
    
    def multiclass_brier_score(self, Y_true, Y_prob):
        # Ensure Y_true is a NumPy array of integers
        Y_true = np.array(Y_true, dtype=int)
        
        # Convert Y_prob to a 2D NumPy array if it's not already
        Y_prob = np.array(Y_prob)
        n_classes = len(np.unique(Y_true))  # Determine the number of classes
        Y_prob_2d = np.zeros((len(Y_true), n_classes))  # Initialize a 2D array for probabilities

        # Assign the predicted probabilities to the correct class index
        for i in range(len(Y_true)):
            class_index = Y_true[i]
            if class_index < n_classes:  # Ensure the index is within the expected range
                Y_prob_2d[i, class_index] = Y_prob[i]

        brier_scores = []
        for class_label in range(n_classes):
            binary_true = (Y_true == class_label).astype(int)  # Binary ground truth for each class
            binary_prob = Y_prob_2d[:, class_label]  # Predicted probabilities for each class
            score = brier_score_loss(binary_true, binary_prob)
            brier_scores.append(score)
        
        return np.mean(brier_scores)


    def auc_multiclass(self,Y_true, Y_prob):
        Y_prob = np.array(Y_prob).reshape(-1, 1)
        Y_prob = np.hstack((1 - Y_prob, Y_prob * 0.5, Y_prob))
        Y_true_binarized = label_binarize(Y_true, classes=[0, 1, 2])

        return roc_auc_score(Y_true_binarized, Y_prob, average="macro", multi_class="ovr")

    # prints model performance according to the following metrics:
    # balanced accuracy, F1 score, AUC, Brier score, KL divergence,
    # and training and test times. But note that training time is
    # dependent on model training externally to this program, which
    # is the case of Bayes nets trained via CPT_Generator.py
    def compute_performance(self, Y_true, Y_pred, Y_prob):
        # constant to avoid NAN in KL divergence
        P = np.asarray(Y_true)+0.00001
        # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001

        print(len(Y_true), len(Y_pred), len(Y_prob))
        accuracy = metrics.accuracy_score(Y_true, Y_pred)
        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred, average='weighted')
        # fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        auc = self.auc_multiclass(Y_true, Y_prob)
        brier = self.multiclass_brier_score(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))

        print("\nCOMPUTING performance on test data...")

        print("Accuracy="+str(accuracy))
        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
        print("Training Time=this number should come from the CPT_Generator!")
        print("Inference Time="+str(self.training_time)+" secs.")
        return {
            "accuracy": accuracy,
            "bal_acc": bal_acc,
            "f1_score": f1,
            "auc": auc,
            "brier": brier,
            "kl_div": kl_div,
            "inference_time": self.training_time
        }

    def evaluate_GPR(self, X_test, Y_test, model, likelihood, training_time):
        print("\nEVALUATING Gaussian Process model...")
        test_time = time.time()
        model.eval()
        likelihood.eval()

        Y_true = Y_test
        Y_pred = []
        Y_prob = []
        for values in X_test: # individual predictions
            test_case = torch.Tensor(np.array([values])).to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(test_case))
                pred_mean = float(predictions.mean.item())
                pred_var = float(predictions.variance.item())
                #pred_std = np.sqrt(pred_var)
                #pdf_1 = self.get_gaussian_density_from_std(1, pred_mean, pred_std)
                #pdf_0 = self.get_gaussian_density_from_std(0, pred_mean, pred_std)
                pdf_1 = self.get_gaussian_density_from_var(1, pred_mean, pred_var)
                pdf_0 = self.get_gaussian_density_from_var(0, pred_mean, pred_var)
                prob = pdf_1 / (pdf_1 + pdf_0)
                Y_prob.append(prob)
                Y_pred.append(np.round(prob))
                #if i <= 10: # to print example predictions
                #    print("mean=%.10f var=%.10f y*=%s pred_y=%s" % (pred_mean, pred_var, Y_test.numpy(), prediction))

        test_time = time.time() - test_time
        self.perfromance = self.compute_performance(Y_true, Y_pred, Y_prob)
        print("acc",self.perfromance)
        print("Training Time="+str(training_time)+" secs.")
        print("Test Time="+str(test_time)+" secs.")
        return model, likelihood

    def plot_GPR(self, X_train, Y_train, X_test, Y_test, model, likelihood):
        if self.REDUCE_INPUTS_TO_2D is False: return

        print("Plotting GPR predictions in 3D...")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x1_min, x1_max = X_train.to('cpu')[:, 0].min(), X_train.to('cpu')[:, 0].max()
        x2_min, x2_max = X_train.to('cpu')[:, 1].min(), X_train.to('cpu')[:, 1].max()
        x1_range = np.linspace(x1_min, x1_max, 100)
        x2_range = np.linspace(x2_min, x2_max, 100)        
        X1_test, X2_test = np.meshgrid(x1_range, x2_range)

        X_grid = np.c_[X1_test.ravel(), X2_test.ravel()]
        X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = likelihood(model(X_grid_tensor))
            mean = predictions.mean.cpu().numpy()
        mean = mean.reshape(X1_test.shape)
        ax.plot_surface(X1_test, X2_test, mean, cmap='viridis', alpha=0.5)

        for i in range(len(Y_train)): # plot training data
            ax.scatter(X_train.to('cpu')[i, 0], X_train.to('cpu')[i, 1], Y_train.to('cpu')[i], 
                    color='b' if Y_train[i] == 1 else 'r', s=30, edgecolors='k', label='Training Data' if i == 0 else "")

        for i in range(len(Y_test)): # plot test data
            ax.scatter(X_test[i, 0], X_test[i, 1], Y_test[i], 
                    color='g' if Y_test[i] == 1 else 'y', s=30, edgecolors='k', label='Testing Data' if i == 0 else "")

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('target')
        ax.legend()
        plt.show()
        
    def query_probability(self, query_data):
        query_values_df = pd.DataFrame([query_data])

        # Standardize the query values if needed
        if self.STANDARDISE_DATA:
            query_keys = query_data.keys()
            temp_query = {}
            target_var = self.dataset_column[-1]
            missing_columns = []
            for k in self.dataset_column:
                if k == target_var:
                    continue
                if k not in query_keys:
                    temp_query[k] = 0
                    missing_columns.append(k)
                else:
                    temp_query[k] =  query_data[k]
            
            query_values_df = pd.DataFrame([temp_query])
            query_values_df = pd.DataFrame(self.scaler.transform(query_values_df),
                                        columns=query_values_df.columns)
            
            # if len(missing_columns) > 0:
            #    query_values_df = query_values_df.drop(columns=missing_columns)
            
            # print("missing_columns", missing_columns)

        # Convert the standardized DataFrame to a torch tensor
        query_tensor = torch.Tensor(query_values_df.values).to(self.device)

        # Perform inference
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(query_tensor))
            pred_mean = float(predictions.mean.item())
            pred_var = float(predictions.variance.item())

        # Calculate probabilities for each target class (0, 1, and 2)
        pdf_0 = self.get_gaussian_density_from_var(0, pred_mean, pred_var)
        pdf_1 = self.get_gaussian_density_from_var(1, pred_mean, pred_var)
        pdf_2 = self.get_gaussian_density_from_var(2, pred_mean, pred_var)
        
        # Normalize to obtain a probability distribution
        total_pdf = pdf_0 + pdf_1 + pdf_2
        prob_0 = pdf_0 / total_pdf
        prob_1 = pdf_1 / total_pdf
        prob_2 = pdf_2 / total_pdf

        return {'target=0': prob_0, 'target=1': prob_1, 'target=2': prob_2}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: gpytorch_GPR.py [train_file.csv] [test_file.csv]")
        print("EXAMPLE> gpytorch_GPR.py data_banknote_authentication-train.csv data_banknote_authentication-test.csv")
        exit(0)
    else:
        datafile_train = sys.argv[1]
        datafile_test = sys.argv[2]
        GPR(datafile_train, datafile_test)
