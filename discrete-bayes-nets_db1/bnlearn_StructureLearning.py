import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# choices of scoring functions: bic, k2, bdeu, bds, aic
TRAINING_DATA = 'data\discrete_dementia_data.csv'
# TRAINING_DATA = 'data\dementia_data-MRI-features.csv'
SCORING_FUNCTION = 'bdeu'
MAX_ITERATIONS=2000000000000000000000
VISUALISE_STRUCTURE=True

# data loading using pandas
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

# structure learning using a chosen scoring function (such as 'bic' or 'aic')
model = bn.structure_learning.fit(data, methodtype='hillclimbsearch', scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model=",model)
print("num_model_edges="+str(num_model_edges))

# visualise the learnt structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()