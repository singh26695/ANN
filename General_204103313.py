# Coding Assignment ME-674 Soft Computing
# Name:  -  Manpreet Singh
# Roll No:- 204103313
# Branch:-  Machine Design

# ---------- Model for Training of ANN for Prediction of Power output of Combined Cycle Power Plant ------------------#

import pandas as pa
import numpy as np

# ----> Input Parameters

print("\n Model for Training of ANN \n")
L1 = int(input("Enter the Number of Input Layer Neurons: "))    # L1 -> First Layer (Input Layer) number of neurons
L2 = int(input("Enter the Number of Hidden Layer Neurons: "))    # L2 -> Second Layer (Hidden Layer) number of neurons
L3 = int(input("Enter the Number of Output Layer Neurons: "))     # L3 -> Third Layer (Output Layer) number of neurons
Train_p = int(input("Enter the Number of training pattern to select from training data file: "))       # Number of training pattern from training data file
Test_p = int(input("Enter the Number of Testing pattern to select from training data file: "))         # Number of testing pattern after training pattern from same file

print("\n **  Keep the training data file in same directory and name it as 'Training_data.xlsx' to provide data ** \n")

decision = str(input(" \n is file in same directory ? (Y/N) : "))

if decision == "N":
    print(" Cant Proceed without data ")
    exit(0)
4
learning_rate = float(input("Enter the learning rate: "))
momentum_factor = float(input("Enter the Momentum factor: "))
tolerance = 0.0005
convergence_tol = 1e-7        # The difference btw two successive MSE values as convergence criteria
Max_iteration = 2000   # Maximum number of iteration to perform along with MSE as termination criteria

bias_hidden = 1         # bias value for hidden layer neurons
bias_output = 1         # bias value for output layer neurons

# ========================================= Reading training data file ==========================================

# -> Place input file in same directory
data = np.asarray(pa.read_excel('Training_data.xlsx', usecols=[*range(0, L1+L3)]))
data = data.astype(float)

row_col = np.shape(data)         # -> getting size of input data (row x col)

# Check points for correct data set dimensions
if row_col[1] < L1+L3:
    print(" Insufficient data (columns) for given L1 and L3 no of neurons, Required: " + str(L1+L3))
    exit(1)

if row_col[0] < Train_p + Test_p:
    print(" Insufficient data (rows) for given No of training and testing patterns, Required: " + str(Train_p+Test_p))
    exit(2)

data_train = data[0: Train_p, :]
data_test = data[Train_p: Train_p+Test_p, :]

data_max = [0] * Train_p
data_min = [0] * Train_p

for i in range(Train_p):     # Getting Max and Min value for normalization of training data
    data_max[i] = np.max(data_train[i, :])
    data_min[i] = np.min(data_train[i, :])

for i in range(Train_p):     # -> training data normalization
    for j in range(L1+L3):
        data_train[i][j] = -0.8 + 1.6 * ((data_train[i][j] - data_min[i]) / (data_max[i] - data_min[i]))

# Assigning Input and Target data ->
Training_input = data_train[:, 0: L1]      # -> Training input Matrix taken from xlsx file
Target_output = data_train[:, L1: L1+L2]   # -> Target Output Matrix for given training data

data_max = [0] * Test_p
data_min = [0] * Test_p

for i in range(Test_p):     # Getting Max and Min value for normalization of data for test patterns
    data_max[i] = np.max(data_test[i, :])
    data_min[i] = np.min(data_test[i, :])


for i in range(Test_p):     # -> testing data normalization
    for j in range(L1+L3):
        data_test[i][j] = -0.8 + 1.6 * ((data_test[i][j] - data_min[i]) / (data_max[i] - data_min[i]))

test_input_data = data_test[:, 0: L1]       # -> Test input Data to check model
test_target_data = data_test[:, L1: L1+L2]  # -> Target value of test data for comparison


# ========================================= Reading file ends =============================================

# Random assignment of synaptic weight Matrix,  [V] and [W]

V = np.random.uniform(-1, 1, (L1+1, L2))
W = np.random.uniform(-1, 1, (L2+1, L3))

Input_hidden = [0] * L2   # to store Input for hidden layer
Output_hidden = [0] * L2  # to store Output of hidden layer
Input_output = [0] * L3   # to store Input for Output layer
Output_output = [0] * L3  # to store Output of Output layer

del_w = [[0] * L3 for i in range(L2 + 1)]
del_v = [[0] * L2 for j in range(L1 + 1)]

del_w_momentum = [[0] * L3 for m in range(L2 + 1)]    # for momentum factor -> del_w of iteration t-1
del_v_momentum = [[0] * L2 for n in range(L1 + 1)]
Mean_square_err = 1     # initial value just to enter in while loop
convergence = 1         # to check the convergence of MSE to terminate


def log_sigmoid(x):     # -> log_sigmoid transfer function for output layer
    return 1 / (1 + np.exp(-x))


def tan_sigmoid(x):     # -> log_sigmoid transfer function for hidden layer
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# ---------------------------------- Main Loop Over Mean Square Error -------------------------------#

count = 0

while Mean_square_err > tolerance:
    count += 1

    # assigning del_w and del_v value of previous time step (t-1) iterations
    # ----------------------------------------------------------------------
    for i in range(L3):       # momentum factor for del_w
        for j in range(L2 + 1):
            del_w_momentum[j][i] = del_w[j][i]

    for i in range(L2):        # momentum factor for del_v
        for j in range(L1 + 1):
            del_v_momentum[j][i] = del_v[j][i]

    # setting parameter equal to zero for next iteration
    # ----------------------------------------------------------------------
    Mean_square_err = 0
    del_w = [[0] * L3 for i in range(L2 + 1)]
    del_v = [[0] * L2 for i in range(L1 + 1)]

    # Batch mode of training -> loop over patterns
    # ----------------------------------------------------------------------
    for p in range(len(Training_input)):

        Input = Training_input[p].tolist()
        T = Target_output[p].tolist()
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

        # Error calculation
        temp = 0
        mse = 0
        for i in range(L3):
            temp = T[i] - Output_output[i]
            mse += 0.5 * (temp * temp)
        Mean_square_err += mse / L3

        # Error back propagation for updating synaptic weights
        # -----------------------------------------------------------------
        # Calculating del_w
        for i in range(L2+1):
            for j in range(L3):
                del_w[i][j] += (T[j]-Output_output[j]) * (1 - pow(Output_output[j], 2)) * Output_hidden[i]   # Output_output[j]*(1-Output_output[j])

        Output_hidden.pop(0)

        # Calculating del_v
        for i in range(L1+1):
            for j in range(L2):
                temp = 0
                for k in range(L3):
                    temp += (T[k] - Output_output[k]) * (1 - pow(Output_output[k], 2)) * W[j+1][k]   # Output_output[k] * (1 - Output_output[k] )
                temp = temp / L3
                del_v[i][j] += temp * Output_hidden[j]*(1-Output_hidden[j]) * Input[i]
        Input.pop(0)

    # dividing del_w by number of input pattern
    for i in range(L3):
        for j in range(L2+1):
            del_w[j][i] /= (p+1)

    # dividing del_v by number of input pattern
    for i in range(L2):
        for j in range(L1 + 1):
            del_v[j][i] /= (p+1)

    # ---------------------------------- loop on patterns end ------------------------------

    # Updating synaptic weights
    for i in range(L3):
        for j in range(L2+1):
            W[j][i] += learning_rate * del_w[j][i] + momentum_factor * del_w_momentum[j][i]

    for i in range(L2):
        for j in range(L1+1):
            V[j][i] += learning_rate * del_v[j][i] + momentum_factor * del_v_momentum[j][i]

    Mean_square_err /= (p+1)     # -> dividing Mean Square Error by numbers of pattern (P)

    print("Mean Square Error for " + str(count + 1) + " Iteration :- " + str(Mean_square_err))

    if abs(convergence - Mean_square_err) < convergence_tol:   # to terminate if convergence satisfied
        break
    convergence = Mean_square_err

    if count >= Max_iteration:   # To stop at particular iteration limit
        break

# ------------------------------------ end of while loop over MSE -------------------------------------#

print("\n Mean Square Error after " + str(count) + " iteration :- " + str(Mean_square_err))

Mean_err = 0
print("\n Testing model for given test pattens: \n")

for p in range(len(test_input_data)):

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
        Target[i] = (((Target[i] + 0.8) * (data_max[p] - data_min[p])) / 1.6) + data_min[p]
        Output_output[i] = (((Output_output[i] + 0.8) * (data_max[p] - data_min[p])) / 1.6) + data_min[p]
        Mean_err += abs(Target[i] - Output_output[i])

    print("Predicted Output for " + str(p + 1) + " Test Pattern :- " + str(Output_output))

Mean_err = Mean_err / (p+1)  # Dividing by Test pattern

print("\n Mean Square Error of training =  " + str(Mean_square_err))
print("\n Mean Absolute Prediction Error = " + str(Mean_err))


# print(V)      # to print updated V and W matrix
# print(W)
