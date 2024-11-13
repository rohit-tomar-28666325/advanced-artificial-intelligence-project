import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# choices of scoring functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = 'data\parkinsons_data_clean.csv'
# TRAINING_DATA = 'data\dementia_data-MRI-features.csv'
SCORING_FUNCTION = 'bic'
MAX_ITERATIONS=200000000000000
VISUALISE_STRUCTURE=True

# data loading using pandas
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

# structure learning using a chosen scoring function (such as 'bic' or 'aic')
model = bn.structure_learning.fit(data, methodtype='hillclimbsearch', scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model=",model)
print("num_model_edges="+str(num_model_edges))

# Given edges
from collections import defaultdict

# Step 1: Build a dictionary of children to parents
parents = defaultdict(set)
all_nodes = set()

for parent, child in model['model_edges']:
    parents[child].add(parent)
    all_nodes.update([parent, child])

# Step 2: Construct the P(Node|Parent1, Parent2, ...)
result = []

for node in all_nodes:
    if node in parents:
        parent_list = ",".join(parents[node])
        result.append(f"P({node}|{parent_list})")
    else:
        # Node with no parents
        result.append(f"P({node})")

# Combine the results into the expected format
output = ";".join(result)
print(output)




# visualise the learnt structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()