import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score,
    brier_score_loss
)
from scipy.stats import entropy
from pgmpy.estimators import HillClimbSearch, BDeuScore, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

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
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store results for all metrics
accuracies = []
balanced_accuracies = []
f1_scores = []
roc_aucs = []
brier_scores = []
kl_divergences = []
max_iter = 200000000000

# Cross-validation loop
for train_index, test_index in kf.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Structure learning using Hill Climb and BDeu score
    hc = HillClimbSearch(train_data)
    model = hc.estimate(scoring_method=BDeuScore(train_data), max_iter=max_iter)  # Use BDeu score
    
    # Define Bayesian Network with learned structure and fit CPTs using BayesianEstimator with Laplace smoothing
    bayesian_model = BayesianNetwork(model.edges())
    bayesian_model.fit(train_data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)

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
        evidence = row.drop('status').to_dict()  # Modify target variable if needed
        true_status = row['status']
        
        # Filter evidence to only include values within valid states for each variable
        valid_evidence = {var: value for var, value in evidence.items() if value < n_bins}
        
        # Perform inference to predict 'status'
        try:
            query_result = inference.map_query(variables=['status'], evidence=valid_evidence)
            y_true.append(true_status)
            y_pred.append(query_result['status'])

            try:
                # Perform the probability query
                prob_result = inference.query(variables=['status'], evidence=valid_evidence)
                prob_demented = prob_result.values[1]  # Probability of class 1
                y_prob.append(prob_demented)
            except KeyError as e:
                print(f"KeyError for evidence: {valid_evidence}. Missing key: {e}")
                y_prob.append(0.0000001)  # Append a very small value to avoid NaN
            
        except IndexError:
            continue

    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Filter valid rows without NaNs
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]
    y_prob_clean = y_prob[valid_indices]

    # Calculate metrics for this fold
    unique_labels = set(y_true_clean) | set(y_pred_clean)
    pos_label = 1 if 1 in unique_labels else min(unique_labels)

    accuracies.append(accuracy_score(y_true_clean, y_pred_clean))
    balanced_accuracies.append(balanced_accuracy_score(y_true_clean, y_pred_clean))
    f1_scores.append(f1_score(y_true_clean, y_pred_clean, pos_label=pos_label))
    
    if len(y_prob_clean) > 0:
        roc_aucs.append(roc_auc_score(y_true_clean, y_prob_clean))
        brier_scores.append(brier_score_loss(y_true_clean, y_prob_clean))
        
        # KL Divergence for this fold
        epsilon = 1e-10
        prob_true = np.array([p if y == pos_label else 1 - p for y, p in zip(y_true_clean, y_prob_clean)])
        prob_true = np.clip(prob_true, epsilon, None)
        y_prob_clean = np.clip(y_prob_clean, epsilon, None)

        kl_div = entropy(prob_true, y_prob_clean, base=2)
        kl_divergences.append(kl_div)

# Average the results across folds
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f}")
print(f"ROC AUC: {np.mean(roc_aucs):.4f}")
print(f"Brier Score: {np.mean(brier_scores):.4f}")
print(f"KL Divergence: {np.mean(kl_divergences):.4f}")
