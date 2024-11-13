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
from imblearn.over_sampling import SMOTE


from BayesNetInference import BayesNetInference



class ModelPerfromance:
    discretizer = None
    
    def __init__(self):
        # self.custom_query = "P(status|MDVP:Fo(Hz)=197.076,MDVP:Fhi(Hz)=206.896,MDVP:Flo(Hz)=192.055,MDVP:Jitter(%)=0.00289,MDVP:Jitter(Abs)=0.00001,MDVP:RAP=0.00166,MDVP:PPQ=0.00168,Jitter:DDP=0.00498,MDVP:Shimmer=0.01098,MDVP:Shimmer(dB)=0.097,Shimmer:APQ3=0.00563,Shimmer:APQ5=0.0068,MDVP:APQ=0.00802,Shimmer:DDA=0.01689,NHR=0.00339,HNR=26.775)"
        #self.custom_query = "P(status|MDVP:Fo(Hz)= 162.568,MDVP:Fhi(Hz= 198.346, MDVP:Flo(Hz)=77.63,MDVP:Jitter(%)=0.00502, MDVP:Jitter(Abs)= 0.00003,MDVP:RAP=0.0028, MDVP:PPQ=0.00253, Jitter:DDP=0.00841, MDVP:Shimmer=0.01791, MDVP:Shimmer(dB)=0.168, Shimmer:APQ3=0.00793, Shimmer:APQ5=0.01057,MDVP:APQ=0.01799, Shimmer:DDA=0.0238, NHR=0.0117,HNR=25.678)"
        self.custom_query = " P(status| MDVP:Fo(Hz)=174.688,MDVP:Fhi(Hz)= 240.005, MDVP:Flo(Hz)= 74.287,MDVP:Jitter(%)=0.0136, MDVP:Jitter(Abs)=0.00008,MDVP:RAP=0.00624, MDVP:PPQ=0.00564,Jitter:DDP=0.01873, MDVP:Shimmer=0.02308,MDVP:Shimmer(dB)= 0.256, Shimmer:APQ3=0.01268,Shimmer:APQ5=0.01365, MDVP:APQ= 0.01667,Shimmer:DDA=0.03804, NHR=0.10715, HNR=17.883)"
        self.target = 'status'
        self.dataset_path = "./data/parkinsons_data-og.csv"
        self.model_name = "parkinsons"
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
        data = data.drop(columns=['name'])
        target_data = data[self.target]
       
        # Preprocess data: Handle missing values
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
        kf = KFold(n_splits=7, shuffle=True, random_state=51)
        max_iter = 200000000000
        config_file_path = f"./config/{self.model_name}_cpt.txt"
        dataset_train_path = f"./data/{self.model_name}_train_.csv"
        dataset_test_path = f"./data/{self.model_name}_test_.csv"
            
        
        # Cross-validation loop
        i = 0
        X = self.data.drop(columns=[self.target])  # Features
        y = self.data[self.target]  # Target variable
        for train_index, test_index in kf.split(X):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            
           # Split the data into train and test sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Apply SMOTE to oversample the minority class in the training set
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Check the distribution of classes in the resampled training set
            print("Resampled Training set class distribution:")
            print(y_train_resampled.value_counts(normalize=True))
            
            # Combine features and target for saving the resampled training set
            train_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
            
            
            # `test_data` remains unchanged and can be used as the test set for this fold
            test_data = pd.concat([X_test, y_test], axis=1)
            
   
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
            
            print('train_data count', train_data['status'].value_counts(), len(train_data['status']))           
            print('test_data count', test_data['status'].value_counts(), len(test_data['status']))           
            
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
