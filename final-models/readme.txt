
Before running the code, You may need you install some required dependencies listed below.
    1. torch
    2. gpytorch
    3. numpy
    4. scikit-learn
    5. matplotlib


Each folder contains a main script (main.py) that can be executed to run the model specific 
to that directory.Custom query can be added in in main.py file in self.custom_query array.

Code can be executed using the below command for each model. 

    cmd: python main.py


When you run a custom query it shows the probability of the given target variable like the below output

For target= 1 :  0.6322849962051551  -> this is the result of the first query for the given target
For target= 0 :  0.5480098497784491  -> this is the result of the second query for the given target
