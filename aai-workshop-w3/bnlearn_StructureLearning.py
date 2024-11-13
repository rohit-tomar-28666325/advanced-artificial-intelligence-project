# import bnlearn as bn
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import Counter


# # examples of training data include 'data\lung_cancer-train.csv' or  'data\lang_detect_train.csv', etc.
# # choices of scoring functions: bic, k2, bdeu, bds, aic
# # TRAINING_DATA = 'data\lung_cancer-train.csv'
# TRAINING_DATA = 'data\pp-dementia_data-MRI-features.csv'
# # TRAINING_DATA = 'data\parkinsons_data-VOICE-features.csv'

# SCORING_FUNCTION = 'k2'
# MAX_ITERATIONS = 200000000000  # Adjusted to a more reasonable number
# VISUALISE_STRUCTURE = True
# NUM_RUNS = 1  # Number of runs for structure learning

# # Data loading using pandas
# data = pd.read_csv(TRAINING_DATA, encoding='latin')
# # print("DATA:\n", data)

# # Store edges from each run
# all_model_edges = []

# # Multi-run structure learning
# for i in range(NUM_RUNS):
#     model = bn.structure_learning.fit(data, methodtype='hillclimbsearch', scoretype=SCORING_FUNCTION, max_iter=MAX_ITERATIONS)
#     all_model_edges.extend(model['model_edges'])  # Collect edges from each run

# # Aggregate results: Count the frequency of each edge
# edge_counts = Counter(all_model_edges)

# # Create a new model from the most common edges
# common_edges = edge_counts.most_common()
# threshold = NUM_RUNS // 2  # Change this threshold to adjust the minimum count for an edge to be included
# aggregated_edges = [edge for edge, count in common_edges if count >= threshold]

# # Create the aggregated model
# aggregated_model = {
#     'model_edges': aggregated_edges,
#     'model': None  # Placeholder; you can create a new model here if needed
# }

# # Print results
# num_aggregated_edges = len(aggregated_model['model_edges'])
# print("Aggregated Model Edges:", aggregated_model['model_edges'])
# print("Number of Aggregated Model Edges:", num_aggregated_edges)

# # Visualise the aggregated structure
# if VISUALISE_STRUCTURE:
#     G = nx.DiGraph()
#     G.add_edges_from(aggregated_model['model_edges'])
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(8, 6))
#     nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
#     plt.title('Aggregated Directed Acyclic Graph (DAG)')
#     plt.show()
    
    
    
    
    
    
    
    
    
    
import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Set up the training data and parameters
TRAINING_DATA = 'data/pp-dementia_data-MRI-features.csv'
SCORING_FUNCTION = 'bic'
MAX_ITERATIONS = 200000000000
VISUALISE_STRUCTURE = True

# Load data
data = pd.read_csv(TRAINING_DATA, encoding='latin')
print("DATA:\n", data)

# Define a refined whitelist to include statistically dependent edges for "Group"
whitelist = [
    ('Group', 'Subject_ID'),
    ('Group', 'M/F'),
    ('Group', 'Age'),
    ('Group', 'EDUC'),
    ('Group', 'SES'),
    ('Group', 'MMSE'),
    ('Group', 'CDR'),
    ('Group', 'nWBV')
]

# Structure learning with whitelist and scoring function
model = bn.structure_learning.fit(
    data, 
    methodtype='hillclimbsearch', 
    scoretype=SCORING_FUNCTION, 
    white_list=whitelist,
    bw_list_method='max', 
    max_iter=MAX_ITERATIONS
)

# Display the model edges and total number of edges
num_model_edges = len(model['model_edges'])
print("model=", model)
print("num_model_edges=" + str(num_model_edges))

# Visualise the learnt structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgreen', font_size=10, arrows=True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()
