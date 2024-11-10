import pandas as pd
from sklearn.model_selection import KFold
from gpytorch_GPR import GPR



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
        self.make_query_object()
        self.evaluate_perfromance()

    def clean_column_name(self, column_name):
        column_name = column_name.replace(" ", "")
        column_name = column_name.replace("(", "_")
        column_name = column_name.replace(":", "_")
        column_name = column_name.replace(")", "")
        return column_name

    def make_query_object(self):
        query = self.custom_query.replace(" ", "")
        evedences = query.split("|")[1].split(",")
        query_continuous = {}
        for ev in evedences:
            key, value = ev.split("=")
            key = self.clean_column_name(key)
            query_continuous[key] = value.replace(")", "")

        self.custom_query = query_continuous
        print(query_continuous)

    def load_data(self):
        # Load dataset
        data = pd.read_csv(self.dataset_path)
        data = data.drop(columns=['Subject ID'])
        data = data.drop(columns=['MRI ID'])
        data = data.drop(columns=['Hand'])

        # Preprocess data: Handle missing values
        data['Group'] = data['Group'].map(
            {'Nondemented': 0, 'Demented': 1, 'Converted': 2})
        data['M/F'] = data['M/F'].map({'M': 0, 'F': 1})

        target_data = data[self.target]
        data.fillna(data.median(), inplace=True)

        for col in data.columns:
            k = self.clean_column_name(col)
            data[k] = data[col]
            if (k != col):
                data = data.drop(columns=[col])

        col = [col for col in data.columns if col !=
               self.target] + [self.target]
        data = data[col]

        data[self.target] = target_data
        print(data.head())
        print("Data loaded successfully")
        return data

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
            self.performance_matrix.append(model_eval_obj.perfromance)

            result = model_eval_obj.query_probability(self.custom_query)
            self.custom_query_perfromance.append(result)
            print("Epoch: ", model_eval_obj.NUM_EPOCHS)

        df = pd.DataFrame(self.performance_matrix)
        mean_values = df.mean()
        print(mean_values)

        df = pd.DataFrame(self.custom_query_perfromance)
        mean_values = df.mean()
        print(mean_values)


ModelPerfromance()

#
# accuracy     0.852577
# bal_acc      0.633848
# f1_score     0.809163
# auc          0.933914
# kl_div      24.582848
# dtype: float64
# target=0    0.139564
# target=1    0.679754
# target=2    0.180681
# dtype: float64




# Epoch:  200
# accuracy           0.852577
# bal_acc            0.633848
# f1_score           0.809163
# auc                0.851039
# brier              0.106337
# kl_div            24.533945
# inference_time     2.045066
# dtype: float64
# target=0    0.135522
# target=1    0.685343
# target=2    0.179135