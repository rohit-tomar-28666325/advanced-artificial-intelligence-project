import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import entropy
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore, AICScore
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from gpytorch_GPR import GPR
# from CPT_Generator import CPT_Generator
import os
import pickle

from BayesNetInference import BayesNetInference
from imblearn.over_sampling import SMOTE




class ModelPerfromance:
    discretizer = None
    
    def __init__(self):
        self.custom_query = "P(Group|visit=3,Age=80,EDUC=12,MMSE=22,CDR=0.5,eTIV=1698,nWBV=0.701,ASF=1.034)"
        # self.custom_query = "P(Group|Visit=2,Age=88,EDUC=14,SES=2,MMSE=30,CDR=0,eTIV=2004,nWBV=0.681,ASF=0.876)"
        self.target = 'Group'
        self.dataset_path = "./data/dementia_data_og.csv"
        self.model_name = "dementia"
        self.performance_matrix = []
        self.custom_query_perfromance = []
        self.data = self.load_data()
        self.make_query_discrete()
        
        # self.get_random_varaible()
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
        
        self.custom_query = query_continuous
        print(query_continuous)
        return
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
                                
      
        
        data[self.target] = target_data
        print(data.head())
        print("Data loaded successfully")
        return data
      
                
    
    def evaluate_perfromance(self):
        # Initialize k-fold cross-validation
        
        # kf = KFold(n_splits=5, shuffle=True, random_state=51)
        kf = KFold(n_splits=5, shuffle=True, random_state=51)
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
            
            if os.path.exists(dataset_train_path):
                os.remove(dataset_train_path)
                
            if os.path.exists(dataset_test_path):
                os.remove(dataset_test_path)
            
            #save files 
            train_data.to_csv(dataset_train_path, index=False)
            test_data.to_csv(dataset_test_path, index=False)
            

            
            print("CPT_GENERATED")
            #evaluate_model
            # config_file_path = "./config/parkinsons.txt"
            model_eval_obj = GPR(dataset_train_path, dataset_test_path)
            self.performance_matrix.append(model_eval_obj.perfromance)
            print("QUERY_TESTING")
            
            result = model_eval_obj.query_probability(self.custom_query) 
            self.custom_query_perfromance.append(result)
            
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

# accuracy           0.887423
# bal_acc            0.655870
# f1_score           0.845345
# auc                0.972546
# brier              0.112071
# kl_div            71.216552
# inference_time     0.214870
# dtype: float64
# 0.0    0.575453
# 1.0    0.395493
# 2.0    0.029053
# dtype: float64



# AICScore

# accuracy           0.871423
# bal_acc            0.641584
# f1_score           0.835943
# auc                0.963427
# brier              0.113602
# kl_div            74.247503
# inference_time     0.143103
# dtype: float64
# 0.0    0.600599
# 1.0    0.377203
# 2.0    0.022198
# dtype: float64


# BicScore

# accuracy           0.879315
# bal_acc            0.648833
# f1_score           0.840066
# auc                0.969732
# brier              0.112704
# kl_div            75.538174
# inference_time     0.110945
# dtype: float64
# 0.0    0.564159
# 1.0    0.412848
# 2.0    0.022993
# dtype: float64





# name--MDVP:Fo(Hz)--MDVP:Fhi(Hz)--MDVP:Flo(Hz)--         -Jitter:DDP--MDVP:Shimmer--MDVP:Shimmer(dB)--Shimmer:APQ3--Shimmer:APQ5--MDVP:APQ--Shimmer:DDA--NHR--HNR--status--RPDE--DFA--spread1--spread2--D2--PPE
# phon_R01_S50_4--174.688--240.005--74.287--0.0136--0.00008--0.00624--0.00564--0.01873--0.02308--0.256--0.01268--0.01365--0.01667--0.03804--0.10715--17.883--0--0.407567--0.655683---6.787197--0.158453--2.679772--0.131728
