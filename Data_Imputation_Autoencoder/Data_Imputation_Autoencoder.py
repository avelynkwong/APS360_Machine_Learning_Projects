#Data Imputation using an Autoencoder

import csv
import numpy as np
import random
import torch
import torch.utils.data
import matplotlib.pyplot as plt

import pandas as pd

#analyzing data

header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation',
 'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=header,
    index_col=False)

df.shape # there are 32561 rows (records) in the data frame, and 14 columns (features)

df[:3] # show the first 3 records

subdf = df[["age", "yredu", "capgain", "caploss", "workhr"]]
subdf[:3] # show the first 3 records

np.sum(subdf["caploss"])

#find the min, max, and average for "age", "yredu", "capgain", "caploss", and "workhr"
features = ["age", "yredu", "capgain", "caploss", "workhr"]
max = []
min = []
avg = []
for i in range(len(features)):
  col = df[[features[i]]]
  max.append(int(col.max()))
  min.append(int(col.min()))
  avg.append(float(col.mean()))
  print(f"The max for {features[i]} is {max[i]}")
  print(f"The min for {features[i]} is {min[i]}")
  print(f"The average for {features[i]} is {avg[i]}")
  #normalizing the dataframe
  df[features[i]] = (df[features[i]]-min[i])/(max[i]-min[i])

print(df[:3])

#categorical features
sum(df["sex"] == " Male")

#percentage of male and female records
percent_male = (sum(df["sex"] == " Male")/df.shape[0])*100
percent_female = (sum(df["sex"] == " Female")/df.shape[0])*100

print(f"The percentage of males is {percent_male:.2f}%")
print(f"The percentage of females is {percent_female:.2f}%")

contcols = ["age", "yredu", "capgain", "caploss", "workhr"]
catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]
features = contcols + catcols
df = df[features]

missing = pd.concat([df[c] == " ?" for c in catcols], axis=1).any(axis=1)
df_with_missing = df[missing]
df_not_missing = df[~missing]

num_missing_records = df_with_missing.shape[0]
total_records = df_with_missing.shape[0] + df_not_missing.shape[0]
percent_removed = (num_missing_records/total_records)*100

print(f"The number of missing records is: {num_missing_records}")
print(f"The percentage of records removed is: {percent_removed:.2f}%")

#one-hot encoding

poss_work_values = set(df_not_missing["work"])
print(poss_work_values)

data = pd.get_dummies(df_not_missing)

data[:3]

datanp = data.values.astype(np.float32)
print(np.random.shuffle(datanp))

#dictionaries
cat_index = {}  # Mapping of feature -> start index (column) of feature in a record
cat_values = {} # Mapping of feature -> list of possible categorical values for a feature

# build up the cat_index and cat_values dictionary
for i, header in enumerate(data.keys()):
    if "_" in header: # categorical header
        feature, value = header.split()
        feature = feature[:-1] # remove the last char; it is always an underscore
        if feature not in cat_index:
            cat_index[feature] = i
            cat_values[feature] = [value]
        else:
            cat_values[feature].append(value)

def get_onehot(record, feature):
    """
    Return the portion of `record` that is the one-hot encoding
    of `feature`. For example, since the feature "work" is stored
    in the indices [5:12] in each record, calling `get_range(record, "work")`
    is equivalent to accessing `record[5:12]`.
    
    Args:
        - record: a numpy array representing one record, formatted
                  the same way as a row in `data.np`
        - feature: a string, should be an element of `catcols`
    """
    start_index = cat_index[feature]
    stop_index = cat_index[feature] + len(cat_values[feature])
    return record[start_index:stop_index]


def get_categorical_value(onehot, feature):
    """
    Return the categorical value name of a feature given
    a one-hot vector representing the feature.
    
    Args:
        - onehot: a numpy array one-hot representation of the feature
        - feature: a string, should be an element of `catcols`
        
    Examples:
    
    >>> get_categorical_value(np.array([0., 0., 0., 0., 0., 1., 0.]), "work")
    'State-gov'
    >>> get_categorical_value(np.array([0.1, 0., 1.1, 0.2, 0., 1., 0.]), "work")
    'Private'
    """
    # <----- TODO: WRITE YOUR CODE HERE ----->
    #get index with max value in the onehot array
    max_idx = np.argmax(onehot)
    
    #get the predicted feature
    feature = cat_values[feature][max_idx]

    return feature

# more useful code, used during training, that depends on the function
# you write above

def get_feature(record, feature):
    """
    Return the categorical feature value of a record
    """
    onehot = get_onehot(record, feature)
    return get_categorical_value(onehot, feature)

def get_features(record):
    """
    Return a dictionary of all categorical feature values of a record
    """
    return { f: get_feature(record, f) for f in catcols }

"""#Train/Test Split

Randomly split the data into approximately 70% training, 15% validation and 15% test.
"""

# set the numpy seed for reproducibility
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html
np.random.seed(50)

# randomly shuffle data
# np.random.shuffle() only shuffles the rows so the features are still in order
np.random.shuffle(datanp)

total_examples = datanp.shape[0]
train_data = torch.from_numpy(datanp[:int(0.7*total_examples)])
val_data = torch.from_numpy(datanp[int(0.7*total_examples):int(0.85*total_examples)])
test_data = torch.from_numpy(datanp[int(0.85*total_examples):total_examples])

print(f"The number of training examples is {len(train_data)}")
print(f"The number of validation examples is {len(val_data)}")
print(f"The number of test examples is {len(test_data)}")

#Create Autoencoder Model


from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        self.name  = "autoencoder"
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(57, 32),
            nn.ReLU(),
            nn.Linear(32, 16) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 57), 
            nn.Sigmoid() # get to the range (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

"""#Training"""

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def zero_out_feature(records, feature):
    """ Set the feature missing in records, by setting the appropriate
    columns of records to 0
    """
    start_index = cat_index[feature]
    stop_index = cat_index[feature] + len(cat_values[feature])
    records[:, start_index:stop_index] = 0
    return records

def zero_out_random_feature(records):
    """ Set one random feature missing in records, by setting the 
    appropriate columns of records to 0
    """
    return zero_out_feature(records, random.choice(catcols))

def evaluate(model, dataloader, loss_fxn):
  losses = 0.0
  num_batches = 0

  for data in dataloader:
    datam = zero_out_random_feature(data.clone()) # zero out one categorical feature
            
    ### send stuff to GPU
    if use_cuda and torch.cuda.is_available():
      datam = datam.cuda()
      data = data.cuda()
    ###

    recon = model(datam)

    loss = loss_fxn(recon, data)
    loss.backward()
    num_batches += 1
    losses += loss.item()
  
  total_loss = float(losses) / num_batches
  return total_loss

def train(model, train_loader, valid_loader, batch_size=64, num_epochs=5, learning_rate=1e-4):
    """ Training loop. You should update this."""
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_losses, val_losses, train_acc, val_acc = [], [], [], [], []
    num_iters = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_train_examples = 0
        

        for data in train_loader:
            datam = zero_out_random_feature(data.clone()) # zero out one categorical feature
            
            
            ### send stuff to GPU
            if use_cuda and torch.cuda.is_available():
              datam = datam.cuda()
              data = data.cuda()
            ###

            recon = model(datam)

            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            num_iters += 1
            epoch_loss += loss.item()

        #save training and validation accuracy
        train_losses.append(float(epoch_loss)/num_iters) 
        train_acc.append(get_accuracy(model, train_loader))
        val_losses.append(evaluate(model, valid_loader, criterion))
        val_acc.append(get_accuracy(model, val_loader))
        iters.append(num_iters)

        #print epoch statistics
        print('Epoch:{}, Train Loss:{:.4f}, Val Loss:{:.4f}, Train accuracy:{:.4f}, Validation accuracy:{:.4f}'.format(epoch+1, train_losses[epoch], val_losses[epoch], train_acc[epoch], val_acc[epoch]))

        #save model
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)

    #plot statistics
    # plt.title("Training Curve")
    # plt.plot(iters, train_acc, label="Train")
    # plt.plot(iters, val_acc, label="Validation")
    # plt.xlabel("Iterations")
    # plt.ylabel("Training Accuracy")
    # plt.show()

use_cuda = True

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=128, num_epochs=30, learning_rate=0.03)

# It appears as though the model is stuck at an accuracy btwn 50-60%
# this could be due to averaging over too many training examples,
# I will reduce batch size and learning rate

use_cuda = True

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=64, num_epochs=30, learning_rate=0.01)

# the model performed better and accuracy appears to be increasing steadily
# I will try to increase learning rate to achieve a higher accuracy

use_cuda = True

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

ae_model = AutoEncoder()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=64, num_epochs=30, learning_rate=0.015)

# the model performed worse, and accuracy seems to be stuck at a plateau
# I will try to decrease batch size again and learning rate
# this allows each training example to have more impact in updating the parameters
# also allows for more iterations per epoch

use_cuda = True

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

ae_model = AutoEncoder()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=32, num_epochs=30, learning_rate=0.008)

# Increasing learning rate to increase val accuracy faster
use_cuda = True

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

ae_model = AutoEncoder()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=32, num_epochs=30, learning_rate=0.01)

# It appears that the first hyperparameter re-tuning did performed the best
# I will use the same settings but increase epochs to achieve a higher accuracy

use_cuda = False

ae_model = AutoEncoder()

if use_cuda and torch.cuda.is_available():
  ae_model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle = True)
train(ae_model, train_loader, val_loader, batch_size=64, num_epochs=80, learning_rate=0.01)

"""#Testing"""

def test_net(model, test_loader, batch_size):

  num_batches = 0
  test_loss = 0.0
  loss_fxn = nn.MSELoss()

  for data in test_loader:
    datam = zero_out_random_feature(data.clone()) # zero out one categorical feature
            
    ### send stuff to GPU
    if use_cuda and torch.cuda.is_available():
      datam = datam.cuda()
      data = data.cuda()
    ###

    recon = model(datam)

    loss = loss_fxn(recon, data)
    loss.backward()
    num_batches += 1
    test_loss += loss.item()
  
  test_loss = float(test_loss) / num_batches
  test_acc = get_accuracy(model, test_loader)

  print("Test loss:{:.4f}, Test accuracy:{:.4f}".format(test_loss, test_acc))

best_model = AutoEncoder()
state = torch.load(get_model_name("autoencoder", 64, 0.01, 58))
best_model.load_state_dict(state)

use_cuda = False

if use_cuda and torch.cuda.is_available():
  best_model.cuda()
  print('CUDA is available!  Testing on GPU ...')
else:
  print('CUDA is not available.  Testing on CPU ...')

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
test_net(best_model, test_loader, 64)
