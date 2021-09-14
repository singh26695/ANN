# Coding Assignment ME-674 Soft Computing
# Name:  -  Manpreet Singh
# Roll No:- 204103313
# Branch:-  Machine Design

# ---------- Trained Model for Prediction of Power output of Combined Cycle Power Plant ------------------#

import pandas as pa
import numpy as np

# ----> Predefined Parameters

L1 = 4              # L1 -> First Layer (Input Layer) number of neurons
L2 = 8              # L2 -> Second Layer (Hidden Layer) number of neurons
L3 = 1              # L3 -> Third Layer (Output Layer) number of neurons
Test_p = 50         # Number of testing pattern after training pattern from same file

learning_rate = .6
momentum_factor = .3

bias_hidden = 1         # bias value for hidden layer neurons
bias_output = 1         # bias value for output layer neurons

# ========================================= Reading training data file ==========================================

data = np.asarray(pa.read_excel('testing_data.xlsx', usecols=[*range(0, L1+L3)]))   # -> Place input file in same directory
data = data.astype(float)

row_col = np.shape(data)         # -> getting size of input data (row x col)

# Check points for correct data set dimensions
if row_col[1] < L1+L3:
    print(" Insufficient data (columns) for given L1 and L3 no of neurons, Required: " + str(L1+L3))
    exit(1)

if row_col[0] < Test_p:
    print(" Insufficient data (rows) for given No of training and testing patterns, Required: " + str(Train_p+Test_p))
    exit(2)

data_test = data[0: Test_p, :]

data_max = [0] * Test_p
data_min = [0] * Test_p

for i in range(Test_p):               # Getting Max and Min value for normalization of data for test patterns
    data_max[i] = np.max(data_test[i, :])
    data_min[i] = np.min(data_test[i, :])

for i in range(Test_p):     # -> testing data normalization as per tan_sigmoid output TF (-0.8, 0.8)
    for j in range(row_col[1]):
        data_test[i][j] = -0.8 + 1.6 * ((data_test[i][j] - data_min[i]) / (data_max[i] - data_min[i]))

# Assigning Input and Target data ->

test_input_data = data_test[:, 0: L1]       # -> Test input Data to check model
test_target_data = data_test[:, L1: L1+L2]  # -> Target value of test data for comparison


# ========================================= Reading file ends =============================================

# Optimized synaptic weight Matrix, [V] and [W] over 20000 iteration with MSE = 0.0002 for 1500 training patterns
# MSE = 0.0002

V = [[-1.73687521e-01, -7.99764057e-01, -8.86959543e-01, -1.96104351e-01, -8.21720633e-01, 7.19776930e-01, -7.12536610e-01, 8.37592550e-01],
     [6.62780093e-02, 8.91402398e-01, -8.96694901e-02, 8.78842529e-01, 3.74556639e-01, 5.07363897e-01, -5.90858518e-01, -6.65398561e-01],
     [4.28515082e-01, 4.69208353e-01, 7.03349494e-01, 6.94315032e-01, -4.52683488e-01, -9.81358385e-01, 2.62449269e-01, -8.78916177e-01],
     [6.28208566e-02, 4.54391720e-01, 7.12620858e-01, 7.00201218e-01, 1.13585213e-01, -8.83779327e-01,  1.59984402e-04, -7.72856426e-01],
     [-8.07616734e-01, -1.07790822e-02, 1.10169015e+00, -1.99599254e-01, -8.82016317e-01, 6.15869598e-01, 7.79122806e-01, -8.97142672e-01]]

W = [[ 0.01171191],
     [-0.98536739],
     [-0.22628787],
     [ 0.81058842],
     [ 0.10882348],
     [-0.34160425],
     [ 1.0133952 ],
     [ 0.51886655],
     [-0.21543898]]

# ---------------------------------------------------------------------------------
Input_hidden = [0] * L2   # to store Input for hidden layer
Output_hidden = [0] * L2  # to store Output of hidden layer
Input_output = [0] * L3   # to store Input for Output layer
Output_output = [0] * L3  # to store Output of Output layer


def log_sigmoid(x):     # -> log_sigmoid transfer function for output layer
    return 1 / (1 + np.exp(-x))


def tan_sigmoid(x):     # -> log_sigmoid transfer function for hidden layer
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

Mean_err =0

for p in range(len(test_input_data)):   # -> loop over Testing Patterns

    Input = test_input_data[p].tolist()
    Target = test_target_data[p].tolist()
    Input.insert(0, bias_hidden)    # adding bias value for hidden layer input

    # Forward pass calculation
    # -----------------------------------------------------------------
    for i in range(L2):     # over -> Hidden neurons
        temp = 0
        for j in range(L1+1):       # over -> input neurons + bias
            temp += Input[j] * V[j][i]
        Input_hidden[i] = temp
        Output_hidden[i] = log_sigmoid(Input_hidden[i])
    Output_hidden.insert(0, bias_output)   # adding bias value for output layer input

    for i in range(L3):     # over -> Output neurons
        temp = 0
        for j in range(L2+1):       # over -> Hidden neurons + bias
            temp += Output_hidden[j] * W[j][i]
        Input_output[i] = temp
        Output_output[i] = tan_sigmoid(Input_output[i])

        # De-normalization for output of output neurons
        Output_output[i] = (((Output_output[i] + 0.8) * (data_max[p] - data_min[p])) / 1.6) + data_min[p]
        Target[i] = (((Target[i] + 0.8) * (data_max[p] - data_min[p])) / 1.6) + data_min[p]
        Mean_err += abs(Target[i]-Output_output[i])

    print("Predicted Output for " + str(p+1) + " Pattern :- " + str(Output_output))

Mean_err = Mean_err / (p+1)  # Dividing by Test pattern

print("\n Mean Absolute Prediction Error = " + str(Mean_err))




