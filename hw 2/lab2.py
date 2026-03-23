"""
Lab 2
Preparing the data, computing basic statistics, and constructing simple models are essential steps for data science practice. In this lab, we will try the whole pipeline using PyTorch. Specifically, we will first get familar with the implementation of DNN in PyTorch. Then, we will perform mortality prediction based on the last visit's diagnosis codes using DNN.

Table of Contents:

Nonlinearity
DNN with Pytorch
Assignment
Preprocessing
DNN Model
Some contents of this lab are adapted from Dive into Deep Learning and Official PyTorch Tutorials.

import os
import csv
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
# make dir "deliverables"
DELIVERABLE_PATH = 'deliverables/'
if not os.path.isdir(DELIVERABLE_PATH):
    os.mkdir(DELIVERABLE_PATH)
DATA_PATH = "../LAB2-lib/data"
assert os.path.isdir(DATA_PATH)
!ls {DATA_PATH}
1. Nonlinearity
1.1 Linear Models May Go Wrong
Previously, we implement the linear regression model. However, linear model may sometimes go wrong.

This is because linearity implies the weaker assumption of monotonicity: that any increase in our feature must either always cause an increase in our model’s output (if the corresponding weight is positive), or always cause a decrease in our model’s output (if the corresponding weight is negative).

However, we can easily come up with examples that violate monotonicity. Say for example that we want to predict probability of death based on body temperature. For individuals with a body temperature above 37°C (98.6°F), higher temperatures indicate greater risk. However, for individuals with body temperatures below 37°C, higher temperatures indicate lower risk! In this case too, we might resolve the problem with some clever feature engineering. Namely, we might use the distance from 37°C as our feature.

But what if we want to predict mortality based on diagnosis. It is very hard to perform feature engineering, which requires a lot of domain knowledge.

1.2 Incorporating Hidden Layers
We can overcome these limitations of linear models and handle a more general class of functions by incorporating one or more hidden layers. The easiest way to do this is to stack many fully-connected layers on top of each other. Each layer feeds into the layer above it, until we generate outputs. We can think of the first 𝐿−1
 layers as our representation and the final layer as our linear predictor. This architecture is commonly called a multilayer perceptron, often abbreviated as MLP. Below, we depict an MLP diagrammatically


Formally, this MLP can be expressed as:
𝐇𝐎=𝐗𝐖(1)+𝐛(1),=𝐇𝐖(2)+𝐛(2).
 

1.3 From Linear to Nonlinear
You might be surprised to find out that—in the model defined above—we gain nothing for our troubles! The reason is plain. The hidden units above are given by an affine function of the inputs, and the outputs are just an affine function of the hidden units. An affine function of an affine function is itself an affine function. Moreover, our linear model was already capable of representing any affine function. That is:

𝐎=(𝐗𝐖(1)+𝐛(1))𝐖(2)+𝐛(2)=𝐗𝐖(1)𝐖(2)+𝐛(1)𝐖(2)+𝐛(2)=𝐗𝐖+𝐛.

In order to realize the potential of multilayer architectures, we need one more key ingredient: a nonlinear activation function 𝜎
 to be applied to each hidden unit following the affine transformation. The outputs of activation functions (e.g., 𝜎(⋅))
) are called activations. In general, with activation functions in place, it is no longer possible to collapse our MLP into a linear model:

𝐇𝐎=𝜎(𝐗𝐖(1)+𝐛(1)),=𝐇𝐖(2)+𝐛(2).
 

Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it. They are differentiable operators to transform input signals to outputs, while most of them add non-linearity. Because activation functions are fundamental to deep learning, let us briefly survey some common activation functions.

ReLU Function

The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the rectified linear unit (ReLU). ReLU provides a very simple nonlinear transformation. Given an element 𝑥
, the function is defined as the maximum of that element and 0
:

ReLU(𝑥)=max(𝑥,0).

Informally, the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0. To gain some intuition, we can plot the function. As you can see, the activation function is piecewise linear.

x = torch.arange(-8.0, 8.0, 0.1)
y = torch.relu(x)
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach(), y.detach())
plt.xlabel('x')
plt.ylabel('relu(x)')
Exercise 1 [10 points]
Plot the Tanh Function.

Hint: try torch.tanh().

x = torch.arange(-8.0, 8.0, 0.1)
y = torch.tanh(x)
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach(), y.detach())
plt.xlabel('x')
plt.ylabel('tanh(x)')
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
x = torch.arange(-8.0, 8.0, 0.1)
assert torch.allclose(y, torch.tanh(x), rtol=1e-2)
​
​
2. DNN Model
From a programing standpoint, a DNN model is represented by a class. Any subclass of it must define a forward propagation function that transforms its input into output and must store any necessary parameters. Note that some subclasses do not require any parameters at all. Finally a model must possess a backpropagation function, for purposes of calculating gradients. Fortunately, due to some behind-the-scenes magic supplied by the auto differentiation when defining our own model, we only need to worry about parameters and the forward propagation function.

The following code generates a network with one fully-connected hidden layer with 256 units and ReLU activation, followed by a fully-connected output layer with 10 units (no activation function).

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
​
X = torch.rand(2, 20)
​
net(X)
In this example, we constructed our model by instantiating an nn.Sequential, with layers in the order that they should be executed passed as arguments. In short, nn.Sequential defines a special kind of Module, the class that presents a model in PyTorch. It maintains an ordered list of constituent Modules. Note that each of the two fully-connected layers is an instance of the Linear class which is itself a subclass of Module. The forward propagation (forward) function is also remarkably simple: it chains each block in the list together, passing the output of each as the input to the next. Note that until now, we have been invoking our models via the construction net(X) to obtain their outputs. This is actually just shorthand for net.__call__(X).

A Custom Model

Perhaps the easiest way to develop intuition about how a model works is to implement one ourselves. Before we implement our own custom model, we briefly summarize the basic functionality that each model must provide:

Ingest input data as arguments to its forward propagation function.
Generate an output by having the forward propagation function return a value. Note that the output may have a different shape from the input.
Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation function. Typically this happens automatically.
Store and provide access to those parameters necessary to execute the forward propagation computation.
Initialize model parameters as needed.
In the following snippet, we code up a model from scratch corresponding to an MLP with one hidden layer with 256 hidden units, and a 10-dimensional output layer. Note that the MLP class below inherits the class that represents a model. We will heavily rely on the parent class’s functions, supplying only our own constructor (the __init__ function in Python) and the forward propagation function.

class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer
​
    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(torch.relu(self.hidden(X)))
Let us first focus on the forward propagation function. Note that it takes X as the input, calculates the hidden representation with the activation function applied, and outputs its logits.

We instantiate the MLP’s layers in the constructor and subsequently invoke these layers on each call to the forward propagation function. Note a few key details. First, our customized __init__ function invokes the parent class’s __init__ function via super().__init__() sparing us the pain of restating boilerplate code applicable to most blocks. We then instantiate our two fully-connected layers, assigning them to self.hidden and self.out. Note that unless we implement a new operator, we need not worry about the backpropagation function or parameter initialization. The system will generate these functions automatically. Let us try this out.

net = MLP()
net(X)
Exercise 2 [20 points]
Implement the following model architecture.

Layers	Configuration	Activation Function
fully connected	input size 128, output size 64	ReLU
fully connected	input size 64, output size 32	ReLU
dropout	probability 0.5	-
fully connected	input size 32, output size 1	Sigmoid
"""
TODO: Build the MLP shown above.
HINT: Consider using `nn.Linear`, `nn.Dropout`, `torch.relu`, `torch.sigmoid`.
"""
​
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # DO NOT change the names
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
​
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
model = Net()
​
assert model.fc1.in_features == 128
assert model.fc1.out_features == 64
assert model.fc2.in_features == 64
assert model.fc2.out_features == 32
assert model.fc3.in_features == 32
assert model.fc3.out_features == 1
​
x = torch.rand(2, 128)
output = model.forward(x)
assert output.shape == (2, 1), "Net() is wrong!"
​
​
Assignment [70 points]
In this assignment, you will use MIMIC-III clinical data as raw input to perform Mortality Prediction.

Preprocessing
MIMIC-III is a large, freely-available database comprising deidentified health-related data associated with over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

Due to license issue, we will use the MIMIC-III Demo dataset, which contains all intensive care unit (ICU) stays for 100 patients.

The goal is to extract the diagnosis codes for each admission (model input) and the mortality information (model output).

Patient
This table defines each subject_id in the database, i.e. defines a single patient.

patients = pd.read_csv(os.path.join(f'{DATA_PATH}/PATIENTS.csv'))
print(patients.shape)
patients.head()
Convert date-of-birth to date

Previously, dob is a string. By converting it to date, we can easily calculate the patient age.

patients['dob'] = pd.to_datetime(patients['dob']).dt.date
Prepare mortality label

A valid dod_hosp means that the patient died during an individual hospital admission or ICU stay (label 1).

patients['mortality'] = patients['dod_hosp'].apply(lambda x: 0 if x != x else 1)
Exclude other columns

patients = patients[['subject_id', 'gender', 'dob', 'mortality']]
print(patients.shape)
patients.head()
Admission
This table defines a patient’s hospital admission, hadm_id.

admissions = pd.read_csv(os.path.join(f'{DATA_PATH}/ADMISSIONS.csv'))
print(admissions.shape)
admissions.head()
Convert admittime and dischtime to date

Similar to dob, by converting them to date, we can easily calculate the patient age.

admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
admissions['dischtime'] = pd.to_datetime(admissions['dischtime']).dt.date
Exclude other columns

admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
print(admissions.shape)
admissions.head()
Merge patient and admission info
We will merge the patient and admission table on subject_id.

patients_admissions = pd.merge(admissions, patients, how='inner', on='subject_id')
print(patients_admissions.shape)
patients_admissions.head()
Exclude patients whose age < 18

# calculate admission age
patients_admissions['age'] = patients_admissions.apply(lambda x: (x['admittime'] - x['dob']).days // 365.25, axis=1)
# for patient with age > 89, set it to 89
patients_admissions['age'] = patients_admissions['age'].apply(lambda x: 89 if x > 89 else x)
print("# of admissions with age < 18:", len(patients_admissions[patients_admissions['age'] < 18].groupby('hadm_id')))
print("# of admissions with age >= 89:", len(patients_admissions[patients_admissions['age'] >= 89].groupby('hadm_id')))
patients_admissions = patients_admissions[patients_admissions['age'] >= 18].reset_index(drop=True)
# drop dob column
patients_admissions = patients_admissions.drop(columns='dob')
print(patients_admissions.shape)
patients_admissions.head()
Diagnosis code
This table contains ICD diagnoses for patients, most notably ICD-9 diagnoses.

# set of valid admission ids
valid_adm_ids = set(patients_admissions.hadm_id)
def convert_to_3digit_icd9(dxStr):
    """ convert icd9 to 3-digit version """
    if dxStr.startswith('E'):
        if len(dxStr) > 4: 
            return dxStr[:4]
        else: 
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else: 
            return dxStr
diagnosis_icd = pd.read_csv(os.path.join(f'{DATA_PATH}/DIAGNOSES_ICD.csv'))
print(diagnosis_icd.shape)
diagnosis_icd.head()
print("Original shape:", diagnosis_icd.shape)
Drop invalid admissions

Drop admissions not in valid_adm_ids.

print("# of rows with invalid admissions:", np.count_nonzero(diagnosis_icd['hadm_id'].isin(valid_adm_ids) == False))
diagnosis_icd = diagnosis_icd[diagnosis_icd['hadm_id'].isin(valid_adm_ids)].reset_index(drop=True)
print("Rows with invalid admissions are dropped! Shape:", diagnosis_icd.shape)
Convert to ICD9 3-digit

Since we only have very limited data, converting ICD9 to 3-digit version will make the learning process easier (e.g., the representation will be much smaller).

diagnosis_icd['icd9_3digit'] = diagnosis_icd['icd9_code'].apply(lambda x: convert_to_3digit_icd9(x))
diagnosis_icd.head()
Group by admission

Group ICD9 codes by admission.

diagnosis_icd = diagnosis_icd.groupby('hadm_id')['icd9_3digit'].unique().reset_index()
print(diagnosis_icd.shape)
diagnosis_icd.head()
Merge patient, admission, and diagnosis code info
Merge all three tables together on hadm_id.

df = pd.merge(patients_admissions, diagnosis_icd, how='inner', on='hadm_id')
df
Sort admissions w.r.t. admission time

df = df.sort_values(['subject_id', 'admittime'], ascending=True).reset_index(drop=True)
df.head()
Exclude other columns

df = df[['subject_id', 'gender', 'hadm_id', 'age', 'mortality', 'icd9_3digit']]
df = df.rename(columns={'icd9_3digit': 'icd9'})
df
Statistics
Calculate some statistics.

def mean_max_min_std(series):
    print(f"mean: {np.mean(series):.1f}, min: {np.min(series):.1f}, max: {np.max(series):.1f}, std: {np.std(series):.1f}")
print("Total # of patients:", len(df.groupby('subject_id')), '\n')
​
print("Total # of admissions:", len(df.groupby('hadm_id')), '\n')
​
print(df.groupby(['subject_id', 'gender']).size().groupby('gender').size(), '\n')
​
print("age:")
mean_max_min_std(df.age)
print()
​
print("# of diagnosis codes:")
mean_max_min_std(df['icd9'].dropna().apply(lambda x: len(x)))
print()
    
print("# of admissions:")
mean_max_min_std(df.groupby('subject_id')['hadm_id'].apply(lambda x: len(x.unique())))
print()
    
print(df.groupby(['subject_id', 'mortality']).size().groupby('mortality').size(), '\n')
Convert diagnosis code to index
To make the code machine-recongnizable, we have to convert them from string to index. For example, code '008' will be converted to index 0.

In this way, we can eventaully represent the diagnosis codes within an admission by one-hot vector, which can directly be fed into the model.

Here is detailed introduction to integer and one-hot encodings.

all_codes = list(set([j for i in df.icd9.to_list() for j in i]))
all_codes.sort()
all_codes[:10]
TOTAL_NUM_CODES = len(all_codes)
TOTAL_NUM_CODES
code2idx = {}
for idx, code in enumerate(all_codes):
    code2idx[code] = idx
code2idx
df['icd9'] = df.icd9.apply(lambda x: [code2idx[i] for i in x])
Convert diagnoiss index to str and join by ';'

df['icd9'] = df['icd9'].apply(lambda x: ';'.join([str(i) for i in x]))
df.head()
Train/Test split
We will split the data into 80% training and 20% testing sets. Normally, we should do train/validation/test splits. However, since the data is very limited, we will just do train/test splits for demonstration purpose.

all_patients = list(df.subject_id.unique().tolist())
random.shuffle(all_patients)
train_ids = all_patients[:int(len(all_patients) * 0.8)]
test_ids = all_patients[int(len(all_patients) * 0.8):]
print("# of train:", len(train_ids))
print("# of test:", len(test_ids))
df_train = df[df['subject_id'].isin(train_ids)].reset_index(drop=True)
df_test = df[df['subject_id'].isin(test_ids)].reset_index(drop=True)
Save

df_train.to_csv(os.path.join(f'{DELIVERABLE_PATH}/train.csv'), index=False)
df_test.to_csv(os.path.join(f'{DELIVERABLE_PATH}/test.csv'), index=False)
DNN model
In the previous lab, we implement the linear regression model, which only has one layer. Thanks to the increasing amount of data and growing computing power, deep learning networks tend to be massive with dozens or hundreds of layers, that is where the term "deep" comes from.

You can build one of these deep networks using only weight matrices as we did in the previous problem, but in general it is very cumbersome and difficult to implement. PyTorch has a nice module nn that provides a nice way to efficiently build large neural networks.

Let us get started!

# two helper functions
​
​
def read_csv(filename):
    """ reading csv from filename """
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.DictReader(file, delimiter=',')
        for row in csv_reader:
            data.append(row)
    header = list(data[0].keys())
    return header, data
​
​
def to_one_hot(label, num_class):
    """ convert to one hot label """
    one_hot_label = [0] * num_class
    for i in label:
        one_hot_label[i] = 1
    return one_hot_label
Custom Dataset
First, let us implement a custom dataset using PyTorch class Dataset, which will characterize the key features of the dataset we want to generate. This is similar to the data_iter() function in the previoius lab.

We will use the diagnosis codes as input and mortality as output.

Note that though one patient can have multiple admissions, for this lab, we will only use the diagnosis codes from the last admission since DNN cannot capture the temporal information.

In the following labs, we will try CNN and RNN which can leverage the entire admission sequence and model the temporal dependency.

from torch.utils.data import Dataset
​
​
class CustomDataset(Dataset):
    
    def __init__(self, split):
        # read the csv
        self._df = pd.read_csv(f'{DELIVERABLE_PATH}/{split}.csv')
        # split diagnosis code index by ';' and convert it to integer
        self._df.icd9 = self._df.icd9.apply(lambda x: [int(i) for i in x.split(';')])
        # build data dict
        self._build_data_dict()
        # a list of subject ids
        self._subj_ids = list(self._data.keys())
        # sort the subject ids to maintain a fixed order
        self._subj_ids.sort()
    
    def _build_data_dict(self):
        """ 
        build SUBJECT_ID to ADMISSION dict
            - subject_id
                - icd9: a list of ICD9 code index
                - mortality: 0/1 morality label
        """
        dict_data = {}
        df = self._df.groupby('subject_id').agg({'mortality': lambda x: x.iloc[0], 'icd9': list}).reset_index()
        for idx, row in df.iterrows():
            subj_id = row.subject_id
            dict_data[subj_id] = {}
            dict_data[subj_id]['icd9'] = row.icd9
            dict_data[subj_id]['mortality'] = row.mortality
        self._data = dict_data
    
    def __len__(self):
        """ return the number of samples (i.e. patients). """
        return len(self._subj_ids)
    
    def __getitem__(self, index):
        """ generates one sample of data. """
        # obtain the subject id
        subj_id = self._subj_ids[index]
        # obtain the data dict by subject id
        data = self._data[subj_id]
        # convert last admission's diagnosis code index to one hot
        x = torch.tensor(to_one_hot(data['icd9'][-1], TOTAL_NUM_CODES), dtype=torch.float32)
        # mortality label
        y = torch.tensor(data['mortality'], dtype=torch.float32)
        return x, y
train_dataset = CustomDataset('train')
test_dataset = CustomDataset('test')
print('Size of training set:', len(train_dataset))
print('Size of testing set:', len(test_dataset))
Here is an example of 𝑥
, and 𝑦
.

Note that 𝑥
 is of shape 271
, which means there are 271
 diagnosis codes in total. It is in one-hot format. A 1
 in position 𝑖
 means that diagnosis code of index 𝑖
 appears in the last admission.

And 𝑦
 is either 0
 or 1
.

x, y = train_dataset[0]
print(f'Example x (shape {x.shape}):\n', x)
print(f'Example y:\n', y)
Next, we will load the dataset into a dataloader so that we can we can use it to loop through the dataset for training and testing.

from torch.utils.data import DataLoader
​
# how many samples per batch to load
batch_size = 8
​
# prepare dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
​
print("# of train batches:", len(train_loader))
print("# of test batches:", len(test_loader))
You will notice that the data loader is created with a batch size of 8
, and shuffle=True.

The batch size is the number of samples we get in one iteration from the data loader and pass through our network, often called a batch.

And shuffle=True tells it to shuffle the dataset every time we start going through the data loader again.

train_iter = iter(train_loader)
x, y = next(train_iter)
​
print('Shape of a batch x:', x.shape)
print('Shape of a batch y:', y.shape)
Build the Model [30 points]
Now, let us build a real NN model. For each patient, the NN model will take an input tensor of 271-dim, and produce an output tensor of 1-dim (0 for non-mortality, 1 for moratality). The detailed model architecture is shown in the table below.

Layers	Configuration	Activation Function	Output Dimension (batch, feature)
fully connected	input size 271, output size 16	ReLU	(8, 16)
dropout	probability 0.5	-	(8, 16)
fully connected	input size 16, output size 1	Sigmoid	(8, 1)
"""
TODO: Build the MLP shown above.
HINT: Consider using `nn.Linear`, `nn.Dropout`, `torch.relu`, `torch.sigmoid`.
"""
​
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # DO NOT change the names
        self.fc1 = nn.Linear(271, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)
​
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
# initialize the NN
model = Net()
print(model)
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
model = Net()
​
assert model.fc1.in_features == 271
assert model.fc1.out_features == 16
assert model.fc2.in_features == 16
assert model.fc2.out_features == 1
​
train_iter = iter(train_loader)
x, y = next(train_iter)
output = model.forward(x)
assert output.shape == (8, 1), "Net() is wrong!"
​
​
Now that we have a network, let's see what happens when we pass in some data.

model = Net()
​
# Grab some data 
train_iter = iter(train_loader)
x, y = next(train_iter)
​
# Forward pass through the network
output = model.forward(x)
​
print('Input x shape:', x.shape)
print('Output shape: ', output.shape)
Train the Network [40 points]
In this step, you will train the NN model.

Neural networks with non-linear activations work like universal function approximators. There is some function that maps your input to the output. The power of neural networks is that we can train them to approximate this function, and basically any function given enough data and compute time.

model = Net()
Losses in PyTorch [10 points]

In the previous lab, we implement the loss function from scratch.

Let us start by seeing how we calculate the loss with PyTorch. Through the nn.module, PyTorch provides losses such as the binary cross-entropy loss (nn.BCELoss). You will usually see the loss assigned to criterion.

As noted in the last part, with a classification problem such as Mortality Prediction, we are using the Sigmoid function to predict mortality probability. With a Sigmoid output, you want to use binary cross-entropy as the loss. To actually calculate the loss, you first define the criterion then pass in the output of your network and the correct labels.

"""
TODO: Define the loss (BCELoss), assign it to `criterion`.
​
REFERENCE: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
"""
​
criterion = nn.BCELoss()
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
assert type(criterion) is nn.modules.loss.BCELoss, "criterion is not BCELoss!"
​
​
Optimizer in PyTorch [10 points]

Optimizer can update the weights with the gradients. In the previous lab, we implement the sgd optimizer from scratch. We can get these from PyTorch's optim package. For example we can use stochastic gradient descent with optim.SGD.

"""
TODO: Define the optimizer (SGD) with learning rate 0.01, assign it to `optimizer`.
​
REFERENCE: https://pytorch.org/docs/stable/optim.html
"""
​
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
assert type(optimizer) is torch.optim.SGD, "optimizer is not SGD!"
assert optimizer.param_groups[0]['lr'] == 0.01, "learning rate is not 0.01!"
​
​
Now let us train the NN model we previously created.

Evaluate [10 points]

First, let us implement the evaluate function that will be called to evaluate the model performance when training.

from sklearn.metrics import *
​
#input: Y_score,Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_score, Y_pred, Y_true):
    acc, auc, precision, recall, f1score = accuracy_score(Y_true, Y_pred), \
                                           roc_auc_score(Y_true, Y_score), \
                                           precision_score(Y_true, Y_pred), \
                                           recall_score(Y_true, Y_pred), \
                                           f1_score(Y_true, Y_pred)
    return acc, auc, precision, recall, f1score
​
​
#input: model, loader
def evaluate(model, loader):
    model.eval()
    all_y_true = torch.LongTensor()
    all_y_pred = torch.LongTensor()
    all_y_score = torch.FloatTensor()
    for x, y in loader:
        # pass the input through the model
        y_hat = model(x)
        # convert shape from [batch size, 1] to [batch size]
        y_hat = y_hat.view(y_hat.shape[0])
        """
        TODO: obtain the predicted class (0, 1) by comparing y_hat against 0.5,
        assign the predicted class to y_pred.
        """
        y_pred = (y_hat >= 0.5).long()
        all_y_true = torch.cat((all_y_true, y.to('cpu')), dim=0)
        all_y_pred = torch.cat((all_y_pred,  y_pred.to('cpu')), dim=0)
        all_y_score = torch.cat((all_y_score,  y_hat.to('cpu')), dim=0)
        
    acc, auc, precision, recall, f1 = classification_metrics(all_y_score.detach().numpy(), 
                                                             all_y_pred.detach().numpy(), 
                                                             all_y_true.detach().numpy())
    print(f"acc: {acc:.3f}, auc: {auc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
    return acc, auc, precision, recall, f1
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
print("model perfomance before training:")
acc, auc, precision, recall, f1 = evaluate(model, train_loader)
assert auc <= 0.6
acc, auc, precision, recall, f1 = evaluate(model, test_loader)
assert auc <= 0.6
​
​
Training [10 points]

To train the model, you should follow the following step:

Clear the gradients of all optimized variables
Forward pass: compute predicted outputs by passing inputs to the model
Calculate the loss
Backward pass: compute gradient of the loss with respect to model parameters
Perform a single optimization step (parameter update)
Update average training loss
# number of epochs to train the model
# feel free to change this
n_epochs = 60
​
# prep model for training
model.train()
​
for epoch in range(n_epochs):
    
    train_loss = 0
    for x, y in train_loader:
        """ Step 1. clear gradients """
        optimizer.zero_grad()
        """ 
        TODO: Step 2. perform forward pass using `model`, save the output to y_hat;
              Step 3. calculate the loss using `criterion`, save the output to loss.
        """
        y_hat = model(x).view(-1)
        loss = criterion(y_hat, y)
        """ Step 4. backward pass """
        loss.backward()
        """ Step 5. optimization """
        optimizer.step()
        """ Step 6. record loss """
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    evaluate(model, test_loader)
'''
AUTOGRADER CELL. DO NOT MODIFY THIS.
'''
​
acc, auc, precision, recall, f1 = evaluate(model, test_loader)
assert auc > 0.6
​
​
You should get a auc score around  0.6
 . This is not ideal since we only have very limited amount of data. With more data, we will expect much better performance (usually with a larger model).
"""
