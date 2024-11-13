import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import entropy
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore, AICScore
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from CPT_Generator import CPT_Generator
from ModelEvaluator import ModelEvaluator
import os
import pickle

from BayesNetInference import BayesNetInference



class ModelPerfromance:
    discretizer = None
    
    def __init__(self):
        # self.custom_query = "P(Group|visit=2,Age=88,EDUC=14,SES=2,MMSE=30,CDR=0,eTIV=2004,nWBV=0.681,ASF=0.876)"
        self.custom_query = "P(Group|Visit=3,Age=80,EDUC=12,MMSE=22,CDR=0.5,eTIV=1698,nWBV=0.701,ASF=1.034)"
        self.target = 'Group'
        self.dataset_path = "./data/dementia_data_og.csv"
        self.model_name = "dementia"
        self.performance_matrix = []
        self.custom_query_perfromance = []
        self.data = self.load_data()
        self.make_query_discrete()
        
        self.get_random_varaible()
        self.evaluate_perfromance()

    def clean_column_name(self,column_name):
        column_name = column_name.replace(" ","")
        column_name = column_name.replace("(","_")
        column_name = column_name.replace(":","_")
        column_name = column_name.replace(")","")
        return column_name
    
    def get_random_varaible(self):
        rand_vars = []
        i=0
        for k in self.data.columns:
            k = self.clean_column_name(k)
            rand_vars.append(f'X{i}({k})')
            i+=1
        self.random_var = ";".join(rand_vars)
        print("Random_variable",self.random_var)
        
    
    def create_cpt_file_structure(self,file_path, model_edges):
        structure = self.create_structure_from_edges(model_edges)
        print("elf.random_var",self.random_var)
        with open(file_path, 'w') as cfg_file:
            cfg_file.write("name:"+str(self.model_name))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("random_variables:"+str(self.random_var))
            cfg_file.write('\n')
            cfg_file.write('\n')
            cfg_file.write("structure:"+str(structure))
            cfg_file.write('\n')
            cfg_file.write('\n')
            
    def make_query_discrete(self):
        query = self.custom_query.replace(" ", "")
        target = query.split("|")[0].replace("P(", "")
        evedences = query.split("|")[1].split(",")

        query_continuous = {}
        for ev in evedences:
            key, value = ev.split("=")
            key = self.clean_column_name(key)
            query_continuous[key] = value.replace(")", "")
        
        
        # Create a DataFrame with a single row for the discretizer
        query_df = pd.DataFrame([query_continuous], columns=self.data.columns).fillna(0)

        # Transform the continuous query values to discrete bins
        transformed = self.discretizer.transform(query_df)  # Shape (1, n_features)

        # Create a new dictionary to hold discrete values
        query_discrete = query_continuous.copy()
        for i, col in enumerate(self.data.columns):
            if col in query_discrete:
                query_discrete[col] = int(transformed[0, i])

        ev = [f'{key}={query_discrete[key]}' for key in query_discrete]
        print(f'P({target}|{",".join(ev)})')
        self.custom_query =  f'P({target}|{",".join(ev)})'
    

        # # Create a DataFrame with a single row for the discretizer
        # query_df = pd.DataFrame([query_continuous], columns=continuous_columns).fillna(0)
        
        # print('query_df',query_df)

        # # Transform the continuous query values to discrete bins
        # transformed = discretizer.transform(query_df)  # Shape (1, n_features)
        # print("transformed",transformed)
        # # Create a new dictionary to hold discrete values
        # query_discrete = query_continuous.copy()
        # for i, col in enumerate(continuous_columns):
        #     if col in query_discrete:
        #         query_discrete[col] = int(transformed[0, i])

        # print('query_discrete',query_discrete)
        # replaced_key_data = {}
        # for key in query_discrete:
        #     k = key.replace("(","_").replace(":","_").replace(")","")
        #     replaced_key_data[k] = query_discrete[key]
        # print(replaced_key_data)

        # ev = [f'{key}={replaced_key_data[key]}' for key in replaced_key_data]
        # print(f'P({target}|{",".join(ev)})')
        # return f'P({target}|{",".join(ev)})'
            
    
    def load_data(self):
        # Load dataset
        data = pd.read_csv(self.dataset_path)
        data = data.drop(columns=['Subject ID'])
        data = data.drop(columns=['MRI ID'])
        data = data.drop(columns=['Hand'])
       
        # Preprocess data: Handle missing values
        data['Group'] = data['Group'].map({'Nondemented': 0, 'Demented': 1, 'Converted':2})
        data['M/F'] = data['M/F'].map({'M': 0, 'F': 1})
        
        target_data = data[self.target]              
        data.fillna(data.median(), inplace=True)
        
        for col in data.columns:
            k = self.clean_column_name(col)
            data[k] = data[col]
            if(k != col):
                data = data.drop(columns=[col])
        
        col = [col for col in data.columns if col != self.target] + [self.target]
        data = data[col]        
                                
        # Discretize continuous variables (example with 4 bins for each feature)
        self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform', subsample=None)
        data[data.columns] = self.discretizer.fit_transform(
            data[data.columns])
        
        # n_bins = 4
        # for col in data.columns:
        #     if data[col].dtype != 'object':
        #         data[col] = pd.cut(data[col], bins=n_bins, labels=False)

        # Encode categorical columns if any (convert to integer categories)
        for col in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        data[self.target] = target_data
        data= data.astype(int)
        
        # with open('./config/discretizer.pkl', 'wb') as f:
        #   pickle.dump(self.discretizer, f)
                
        print("Data loaded successfully")
        return data
        
        
    def create_structure_from_edges(self,model_edges):
        parents = defaultdict(set)
        all_nodes = set()

        for parent, child in model_edges:
            parents[child].add(parent)
            all_nodes.update([parent, child])

        result = []
        for node in all_nodes:
            if node in parents:
                parent_list = ",".join(parents[node])
                result.append(f"P({node}|{parent_list})")
            else:
                result.append(f"P({node})")

        output = ";".join(result)
        print("Model Structure-> ",output)
        return output
                
    
    def evaluate_perfromance(self):
        # Initialize k-fold cross-validation
        
        # kf = KFold(n_splits=5, shuffle=True, random_state=51)
        kf = KFold(n_splits=5, shuffle=True, random_state=51)
        max_iter = 200000000000
        config_file_path = f"./config/{self.model_name}_cpt.txt"
        dataset_train_path = f"./data/{self.model_name}_train_.csv"
        dataset_test_path = f"./data/{self.model_name}_test_.csv"
            
        
        # Cross-validation loop
        i = 0
        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            
   
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
            
            if os.path.exists(dataset_train_path):
                os.remove(dataset_train_path)
                
            if os.path.exists(dataset_test_path):
                os.remove(dataset_test_path)
            
            #save files 
            train_data.to_csv(dataset_train_path, index=False)
            test_data.to_csv(dataset_test_path, index=False)
            

            
            # Structure learning using Hill Climb and BDeu score
            hc = HillClimbSearch(train_data)
            model = hc.estimate(scoring_method=BicScore(train_data), max_iter=max_iter)  # Use BDeu score
            
            # config_file_path = "./config/parkinsons.txt"
            self.create_cpt_file_structure(config_file_path, model.edges())
            
            # create cpt for defiend structure
            print("CPT_GENERATION_START")
            
            CPT_Generator(config_file_path, dataset_train_path)
            
            print("CPT_GENERATED")
            #evaluate_model
            # config_file_path = "./config/parkinsons.txt"
            model_eval_obj = ModelEvaluator(config_file_path, dataset_test_path)
            self.performance_matrix.append(model_eval_obj.performance_matrix)
            alg_name = "InferenceByEnumeration"
            bni = BayesNetInference(alg_name, config_file_path, self.custom_query,None) 
            self.custom_query_perfromance.append(bni.normalised_dist)
            
            print('train_data count', train_data[self.target].value_counts(), len(train_data[self.target]))           
            print('test_data count', test_data[self.target].value_counts(), len(test_data[self.target]))           
            
        # print(self.performance_matrix)
        df = pd.DataFrame(self.performance_matrix)
        # Calculate the mean for each column
        mean_values = df.mean()
        print(mean_values)
        
        
        df = pd.DataFrame(self.custom_query_perfromance)
        # Calculate the mean for each column
        mean_values = df.mean()
        print(mean_values)
        
        
        
        
    
 
ModelPerfromance()



# BDeuScore

# accuracy          0.846154
# bal_acc           0.751292
# f1_score          0.838384
# auc               0.883508
# brier             0.084752
# kl_div            6.201897
# inference_time    3.521707



# AICScore

# accuracy          0.866667
# bal_acc           0.778673
# f1_score          0.858670
# auc               0.877023
# brier             0.078215
# kl_div            6.014883
# inference_time    2.774548


# BicScore

# accuracy          0.856410
# bal_acc           0.795110
# f1_score          0.852603
# auc               0.869215
# brier             0.078236
# kl_div            5.862516
# inference_time    2.540262





# name--MDVP:Fo(Hz)--MDVP:Fhi(Hz)--MDVP:Flo(Hz)--         -Jitter:DDP--MDVP:Shimmer--MDVP:Shimmer(dB)--Shimmer:APQ3--Shimmer:APQ5--MDVP:APQ--Shimmer:DDA--NHR--HNR--status--RPDE--DFA--spread1--spread2--D2--PPE
# phon_R01_S50_4--174.688--240.005--74.287--0.0136--0.00008--0.00624--0.00564--0.01873--0.02308--0.256--0.01268--0.01365--0.01667--0.03804--0.10715--17.883--0--0.407567--0.655683---6.787197--0.158453--2.679772--0.131728
