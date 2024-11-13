import pandas as pd
from sklearn.model_selection import KFold
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore, AICScore, PC
from collections import defaultdict
from PDF_Generator import PDF_Generator
from ModelEvaluator import ModelEvaluator
from BayesNetInference import BayesNetInference
import os




class ModelPerfromance:
    def __init__(self):
        self.custom_query = [
            "P(status=0|MDVP:Fo(Hz)=197.076,MDVP:Fhi(Hz)=206.896,MDVP:Flo(Hz)=192.055,MDVP:Jitter(%)=0.00289,MDVP:Jitter(Abs)=0.00001,MDVP:RAP=0.00166,MDVP:PPQ=0.00168,Jitter:DDP=0.00498,MDVP:Shimmer=0.01098,MDVP:Shimmer(dB)=0.097,Shimmer:APQ3=0.00563,Shimmer:APQ5=0.0068,MDVP:APQ=0.00802,Shimmer:DDA=0.01689,NHR=0.00339,HNR=26.775)",
            "P(status=1|MDVP:Fo(Hz)=162.568,MDVP:Fhi(Hz=198.346,MDVP:Flo(Hz)=77.63,MDVP:Jitter(%)=0.00502,MDVP:Jitter(Abs)=0.00003,MDVP:RAP=0.0028,MDVP:PPQ=0.00253,Jitter:DDP=0.00841,MDVP:Shimmer=0.01791,MDVP:Shimmer(dB)=0.168,Shimmer:APQ3=0.00793,Shimmer:APQ5=0.01057,MDVP:APQ=0.01799,Shimmer:DDA=0.0238,NHR=0.0117,HNR=25.678)"
        ]
        self.clean_query = {}
        self.target = 'status'
        self.dataset_path = "./data/parkinsons_data-og.csv"
        self.model_name = "parkinsons"
        self.performance_matrix = []
        self.custom_query_perfromance = []
        self.data = self.load_data()
        self.clean_user_query()
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
            
    def clean_user_query(self):
        index = 0
        for query in self.custom_query:
            query = query.replace(" ", "")
            target = query.split("|")[0].replace("P(", "").split("=")[1]
            evedences = query.split("|")[1].split(",")

            query_continuous = {}
            for ev in evedences:
                key, value = ev.split("=")
                key = self.clean_column_name(key)
                query_continuous[key] = value.replace(")", "")
        
            ev = [f'{key}={query_continuous[key]}' for key in query_continuous]
            self.clean_query[str(index)] = {
                'target_val': target,
                "query": f'P({self.target}|{",".join(ev)})',
                "result": []
            }
            index += 1
            
        
        print(self.clean_query)
            
    
    
    def load_data(self):
        # Load dataset
        data = pd.read_csv(self.dataset_path)
        data = data.drop(columns=['name'])
        target_data = data[self.target]
        data.fillna(data.median(), inplace=True)
        
        for col in data.columns:
            k = self.clean_column_name(col)
            data[k] = data[col]
            if k != col:
                data = data.drop(columns=[col])
        
        col = [col for col in data.columns if col != self.target] + [self.target]
        data = data[col]        
        data[self.target] = target_data
        print(data.head())
        print("Data loaded and standardized successfully")
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
        kf = KFold(n_splits=5, shuffle=True, random_state=51)
        max_iter = 20000000000000
        config_file_path = f"./config/{self.model_name}_cpt.txt"
        dataset_train_path = f"./data/{self.model_name}_train_.csv"
        dataset_test_path = f"./data/{self.model_name}_test_.csv"
            
        
        # Cross-validation loop
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
            
         
            # Structure learning using Hill Climb
            hc = HillClimbSearch(train_data)
            model = hc.estimate(scoring_method=BDeuScore(train_data), max_iter=max_iter)
             
            self.create_cpt_file_structure(config_file_path, model.edges())
            
            pdfObj = PDF_Generator(config_file_path, dataset_train_path)           
            model_eval_obj = ModelEvaluator(
                config_file_path, dataset_test_path)
            self.performance_matrix.append({**model_eval_obj.performance_matrix, "training_time": pdfObj.running_time})

            alg_name = "InferenceByEnumeration"
            index = 0
            for key in self.clean_query:
                bni = BayesNetInference(alg_name, config_file_path, self.clean_query[key]['query'], None)
                self.clean_query[str(index)]['result'].append(bni.normalised_dist)
                index+=1
            
            print('train_data count', train_data[self.target].value_counts(), len(train_data[self.target]))           
            print('test_data count', test_data[self.target].value_counts(), len(test_data[self.target]))           
            
        
        df = pd.DataFrame(self.performance_matrix)
        mean_values = df.mean()
        print(mean_values)
        
        for key in self.clean_query:
            target = self.clean_query[key]['target_val']
            df = pd.DataFrame(self.clean_query[key]['result'])
            mean_values = df.mean()
            print("For target= "+ target + " : ",  mean_values[float(target)])
    
 
ModelPerfromance()
