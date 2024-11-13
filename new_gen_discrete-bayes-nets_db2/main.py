import pandas as pd
import pickle
import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score,
    brier_score_loss
)
from scipy.stats import entropy
from pgmpy.estimators import HillClimbSearch, BDeuScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork


def get_random_variable(data):
    s="MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE"
    s = s.split(",")
    rand_vars = ""
    i=0
    for k in s:
        k = k.replace("(","_")
        k = k.replace(":","_")
        k = k.replace(")","")
        rand_vars+= f'X{i}({k});'
        i+=1
        
    return rand_vars

def create_cpt_file(name, file_path, structure):
    rand_vars = []
    rand_vars = str(rand_vars).replace('[', '').replace(']', '')
    rand_vars = str(rand_vars).replace('\'', '').replace(', ', ';')

    structure = []
    structure = str(structure).replace('[', '').replace(']', '')
    structure = str(structure).replace('\'', '').replace(', ', ';')

    with open(file_path, 'w') as cfg_file:
        cfg_file.write("name:"+str(name))
        cfg_file.write('\n')
        cfg_file.write('\n')
        cfg_file.write("random_variables:"+str(rand_vars))
        cfg_file.write('\n')
        cfg_file.write('\n')
        cfg_file.write("structure:"+str(structure))
        cfg_file.write('\n')
        cfg_file.write('\n')
        # for key, cpt in self.CPTs.items():
        #     cpt_header = key.replace("P(", "CPT(")
        #     cfg_file.write(str(cpt_header)+":")
        #     cfg_file.write('\n')
        #     num_written_probs = 0
        #     for domain_vals, probability in cpt.items():
        #         num_written_probs += 1
        #         line = str(domain_vals)+"="+str(probability)
        #         line = line+";" if num_written_probs < len(cpt) else line
        #         cfg_file.write(line)
        #         cfg_file.write('\n')
        #     cfg_file.write('\n')




# Load dataset
data = pd.read_csv('./parkinsons_data-og.csv')

data = data.drop(columns=['name'])

# Preprocess data: Handle missing values
data.fillna(data.median(), inplace=True)

# Discretize continuous variables (example with 4 bins for each feature)
n_bins = 4
for col in data.columns:
    if data[col].dtype != 'object':
        data[col] = pd.cut(data[col], bins=n_bins, labels=False)

# Encode categorical columns if any (convert to integer categories)
from sklearn.preprocessing import LabelEncoder

for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])


# Initialize k-fold cross-validation
kf = KFold(n_splits=8, shuffle=True, random_state=51)

# To store results for all metrics
accuracies = []
balanced_accuracies = []
f1_scores = []
roc_aucs = []
brier_scores = []
kl_divergences = []
max_iter = 200000000000
dataset = "parkinsons"
# Cross-validation loop
i = 0
for train_index, test_index in kf.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    
    config_file_path = f"./config/{dataset}_cpt_{i}.txt"
    dataset_train_path = f"./data/{dataset}_train_{i}.csv"
    dataset_test_path = f"./data/{dataset}_test{i}.csv"
    i+=1
    

    
    # Structure learning using Hill Climb and BDeu score
    hc = HillClimbSearch(train_data)
    model = hc.estimate(scoring_method=BDeuScore(train_data), max_iter=max_iter)  # Use BDeu score
    
    # Define Bayesian Network with learned structure and fit CPTs
    bayesian_model = BayesianNetwork(model.edges())
    bayesian_model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    
    # Inference setup
    inference = VariableElimination(bayesian_model)
    with open('./bayesian_model.pkl', 'wb') as f:
        pickle.dump(bayesian_model, f)
    # Lists to store true and predicted values for metrics
    y_true = []
    y_pred = []
    y_prob = []

    # Testing the model
    for _, row in test_data.iterrows():
        # Example: Querying the 'status' variable
        evidence = row.drop('status').to_dict()  # Modify target variable if needed
        true_status = row['status']
        
        # Filter evidence to only include values within valid states for each variable
        valid_evidence = {}
        for var, value in evidence.items():
            if value < n_bins:  # Check if within expected range
                valid_evidence[var] = value
        
        # Perform inference to predict 'status'
        try:
            query_result = inference.map_query(variables=['status'], evidence=valid_evidence,show_progress=False)
            # Store the true and predicted status
            y_true.append(true_status)
            y_pred.append(query_result['status'])

            # try:
                # Perform the probability query
            # print("true_status",true_status)
            # print("valid_evidence",valid_evidence)
            prob_result = inference.query(variables=['status'], evidence=valid_evidence, show_progress=False)
            prob_demented = prob_result.values[1]  # Probability of class 1
            y_prob.append(prob_demented)
            # except KeyError as e:
            #     print(f"KeyError for evidence: {valid_evidence}. Missing key: {e}")
            #     y_prob.append(0.0000001)  # Append a very small value to avoid NaN
            
        except IndexError:
            # Skip this instance if evidence is out of range
            continue

    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    

    # Check for NaN values and filter out any rows with NaNs
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]
    y_prob_clean = y_prob[valid_indices]

    # Check the unique values in y_true_clean and y_pred_clean to determine pos_label
    unique_labels = set(y_true_clean) | set(y_pred_clean)
    pos_label = 1 if 1 in unique_labels else min(unique_labels)

    # Calculate metrics for this fold
    accuracies.append(accuracy_score(y_true_clean, y_pred_clean))
    balanced_accuracies.append(balanced_accuracy_score(y_true_clean, y_pred_clean))
    f1_scores.append(f1_score(y_true_clean, y_pred_clean, pos_label=pos_label))
    
    if len(y_prob_clean) > 0:  # Ensure there are valid probabilities
        roc_aucs.append(roc_auc_score(y_true_clean, y_prob_clean))
        brier_scores.append(brier_score_loss(y_true_clean, y_prob_clean))
        
        # Calculate KL Divergence for this fold
        # Add a small constant to avoid division by zero
        epsilon = 1e-10
        prob_true = np.array([p if y == pos_label else 1 - p for y, p in zip(y_true_clean, y_prob_clean)])
        prob_true = np.clip(prob_true, epsilon, None)  # Ensure no zero probabilities
        y_prob_clean = np.clip(y_prob_clean, epsilon, None)  # Ensure no zero probabilities

        kl_div = entropy(prob_true, y_prob_clean, base=2)
        kl_divergences.append(kl_div)

# Average the results across folds
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f}")
print(f"ROC AUC: {np.mean(roc_aucs):.4f}")
print(f"Brier Score: {np.mean(brier_scores):.4f}")
print(f"KL Divergence: {np.mean(kl_divergences):.4f}")




































# # Load the trained Bayesian model
# with open('bayesian_model.pkl', 'rb') as f:
#     bayesian_model = pickle.load(f)

# # Set up inference
# inference = VariableElimination(bayesian_model)

# # Define your evidence with continuous values
# # for status =1 
# status = 1
# evidence = {
#     'MDVP:Fo(Hz)': 162.568,
#     'MDVP:Fhi(Hz)': 198.346,
#     'MDVP:Flo(Hz)': 77.63,
#     'MDVP:Jitter(%)': 0.00502,
#     'MDVP:Jitter(Abs)': 0.00003,
#     'MDVP:RAP': 0.0028,
#     'MDVP:PPQ': 0.00253,
#     'Jitter:DDP': 0.00841,
#     'MDVP:Shimmer': 0.01791,
#     'MDVP:Shimmer(dB)': 0.168,
#     'Shimmer:APQ3': 0.00793,
#     'Shimmer:APQ5': 0.01057,
#     'MDVP:APQ': 0.01799,
#     'Shimmer:DDA': 0.0238,
#     'NHR': 0.0117,
#     'HNR': 25.678
# }

#for status = 0
# status = 0
# evidence = {
#     'MDVP:Fo(Hz)': 197.076,
#     'MDVP:Fhi(Hz)': 206.896,
#     'MDVP:Flo(Hz)': 192.055,
#     'MDVP:Jitter(%)': 0.00289,
#     'MDVP:Jitter(Abs)': 0.00001,
#     'MDVP:RAP': 0.00166,
#     'MDVP:PPQ': 0.00168,
#     'Jitter:DDP': 0.00498,
#     'MDVP:Shimmer': 0.01098,
#     'MDVP:Shimmer(dB)': 0.097,
#     'Shimmer:APQ3': 0.00563,
#     'Shimmer:APQ5': 0.0068,
#     'MDVP:APQ': 0.00802,
#     'Shimmer:DDA': 0.01689,
#     'NHR': 0.00339,
#     'HNR': 26.775
# }


# # Load your training dataset to get the binning information
# data = pd.read_csv('./parkinsons_data-og.csv')
# data = data.drop(columns=['name'])

# # Define number of bins (should match training configuration)
# n_bins = 4

# # Discretize evidence to match trained model bins
# for col, value in evidence.items():
#     if col in data.columns:
#         # Get bin edges for this column
#         bins = pd.cut(data[col], bins=n_bins, labels=False, retbins=True)[1]
        
#         # Determine which bin the evidence value falls into
#         evidence[col] = np.digitize(value, bins) - 1  # "-1" to zero-index

# # Perform the query for P(status=1 | evidence)
# try:
#     prob_result = inference.query(variables=['status'], evidence=evidence)
#     prob_status_1 = prob_result.values[status]  # Probability of status=1
#     print(f"P(status={status} | evidence) = {prob_status_1:.6f}")
# except KeyError as e:
#     print(f"KeyError: {e}. Please check if evidence is within the trained model's range.")



