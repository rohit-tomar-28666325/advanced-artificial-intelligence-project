import pandas as pd
from sklearn.model_selection import KFold
from gpytorch_GPR import GPR


class ModelPerfromance:
    discretizer = None

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
        self.my_test = []
        
        self.data = self.load_data()
        self.make_query_object()
        self.evaluate_perfromance()

    def clean_column_name(self, column_name):
        column_name = column_name.replace(" ", "")
        column_name = column_name.replace("(", "_")
        column_name = column_name.replace(":", "_")
        column_name = column_name.replace(")", "")
        return column_name

    def make_query_object(self):
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

            self.custom_query = query_continuous
            self.clean_query[str(index)] = {
                'target_val': target,
                "query": query_continuous,
                "result": []
            }
            index += 1

    def load_data(self):
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
        data[self.target] = target_data
        
        print(data.head())
        print("Data loaded successfully")
        return data
    
    def test_dataset_for_specific_target(self,test_data,model_eval_obj, target=0):
        status_0_data = test_data[test_data['status'] == target]
        keys_to_keep = set(self.custom_query.keys())

        # Loop over each row where status == target
        for index, row in status_0_data.iterrows():
            # Convert row to dictionary and keep only specified keys
            filtered_row = {k: row[k] for k in keys_to_keep if k in row}
            result = model_eval_obj.query_probability(filtered_row)
            self.my_test.append(result)
        

    def evaluate_perfromance(self):
        # Initialize k-fold cross-validation

        kf = KFold(n_splits=5, shuffle=True, random_state=51)
        dataset_train_path = f"./data/{self.model_name}_train_.csv"
        dataset_test_path = f"./data/{self.model_name}_test_.csv"

        # Cross-validation loop
        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            # save files
            train_data.to_csv(dataset_train_path, index=False)
            test_data.to_csv(dataset_test_path, index=False)

            # evaluate_model
            model_eval_obj = GPR(dataset_train_path, dataset_test_path)

            result = model_eval_obj.query_probability(self.custom_query)
            
            self.performance_matrix.append({**model_eval_obj.perfromance, "training_time": model_eval_obj.training_time})

            print("QUERY_TESTING")
            index = 0
            for key in self.clean_query:
                result = model_eval_obj.query_probability(self.clean_query[key]['query'])
                self.clean_query[str(index)]['result'].append(result)
                index+=1
            # self.test_dataset_for_specific_target(test_data,model_eval_obj)
            
            self.custom_query_perfromance.append(result)
            print("Epoch: ",model_eval_obj.NUM_EPOCHS)

         

        df = pd.DataFrame(self.performance_matrix)
        mean_values = df.mean()
        print(mean_values)
        
        for key in self.clean_query:
            target = self.clean_query[key]['target_val']
            df = pd.DataFrame(self.clean_query[key]['result'])
            mean_values = df.mean()
            print("For target= "+ target + " : ",  mean_values[target])
        
        
        
        # df = pd.DataFrame(self.my_test)
        # mean_values = df.mean()
        # print(mean_values)
        
        


ModelPerfromance()
