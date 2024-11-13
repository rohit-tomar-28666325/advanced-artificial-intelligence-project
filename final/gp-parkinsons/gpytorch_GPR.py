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
from ModelEvaluator import ModelEvaluator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive


class ExponentialKernel(Kernel):
    def __init__(self, lengthscale=1.0, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.tensor(lengthscale)))
        self.register_constraint("raw_lengthscale", Positive())
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.tensor(variance)))
        self.register_constraint("raw_variance", Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)
    
    def forward(self, x1, x2, diag=False, **params):
        dists = torch.cdist(x1, x2) / (self.lengthscale + 1e-6)
        return self.variance * torch.exp(-dists)


# Gaussian Process regressor using the Radial Basis Function kernel (others possible)
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel()
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5) # matern12
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) # matern32
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5) # matern52
        # self.covar_module = ExponentialKernel(lengthscale=1.0)
        
        self.covar_module = gpytorch.kernels.RQKernel(alpha=1.0)
        # self.covar_module = gpytorch.kernels.PolynomialKernel(2)
        # self.covar_module = gpytorch.kernels.PolynomialKernel(3)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# Gaussian Process regressor applied to classification tasks
class GPR():
    MAX_TRAIN_DATA=2000
    STANDARDISE_DATA = False
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 150
    filename = 'gp_model_and_likelihood.pkl'
    REDUCE_INPUTS_TO_2D = False
    device = None
    pca = None

    def __init__(self, datafile_train, datafile_test):
        self.perfromance = {}
        self.dataset_column = []
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
        self.training_time = training_time
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

        training_time = time.time() - training_time
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
        self.perfromance = ModelEvaluator.compute_performance(None, Y_true, Y_pred, Y_prob)
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
                    temp_query[k] = query_data[k]
            
            query_values_df = pd.DataFrame([temp_query])
            query_values_df = pd.DataFrame(self.scaler.transform(query_values_df), columns=query_values_df.columns)

        # Reindex to match the order of model training features
        query_values_df = query_values_df.reindex(columns=self.dataset_column[:-1], fill_value=0)

        # Ensure all values are numeric
        query_values_df = query_values_df.apply(pd.to_numeric, errors='coerce')

        # Convert the DataFrame to a torch tensor
        # print(query_values_df.values)
        query_tensor = torch.tensor(query_values_df.values, dtype=torch.float32).to(self.device)

        # Perform inference
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(query_tensor))
            pred_mean = float(predictions.mean.item())
            pred_var = float(predictions.variance.item())

        # Calculate probability using Gaussian density
        pdf_1 = self.get_gaussian_density_from_var(1, pred_mean, pred_var)
        pdf_0 = self.get_gaussian_density_from_var(0, pred_mean, pred_var)
        total_pdf = pdf_0 + pdf_1
        prob_0 = pdf_0 / total_pdf
        prob_1 = pdf_1 / total_pdf

        return {'0': prob_0, '1': prob_1}

        


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: gpytorch_GPR.py [train_file.csv] [test_file.csv]")
        print("EXAMPLE> gpytorch_GPR.py data_banknote_authentication-train.csv data_banknote_authentication-test.csv")
        exit(0)
    else:
        datafile_train = sys.argv[1]
        datafile_test = sys.argv[2]
        GPR(datafile_train, datafile_test)
