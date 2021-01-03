# Gesture Recognition using Convolutional Neural Networks

#imports
from google.colab import drive
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

#mount the drive
drive.mount('/content/drive')

#DATA PREPROCESSING

#path to the dataset folder
data_path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/Lab_3b_Gesture_Dataset/'

#create a dataset class that can be used by torch.utils.dataloader to create batches
class datasets(Dataset):
  def __init__(self, mode, data_path):
    super(datasets, self).__init__() #inheriting functions from the Dataset parent class
    self.imgsets, self.labelsets = self.get_data(mode) #get either train, val, or test data

  def get_data(self, mode):
    letter_folders = sorted(os.listdir(data_path)) #make sure the letters are in alphabetical order
    label = 0
    #create lists to store images and labels
    imgset = []
    labelset = []
    #for each letter, add the imgs & corresponding label to the lists
    for letter in letter_folders:
      #imgs is a list of all the img names for the current letter
      imgs = os.listdir(data_path + str(letter))
      num_imgs = len(imgs)
      #set the boundaries for training, validation, and testing data
      if mode == 'train':
        start = 0
        stop = int(0.8*num_imgs)
      if mode == 'val':
        start = int(0.8*num_imgs)
        stop = int(0.9*num_imgs)
      if mode == 'test':
        start = int(0.9*num_imgs)
        stop = num_imgs

      #for each image in the current letter folder, append the img name and label to the lists
      for i in range(start, stop):
        imgset.append(str(letter) + '/' + imgs[i]) #ex. A/1_A_1.jpg
        labelset.append(label)
      #increment the label when you move to the next letter
      label +=1

    #shuffle the imgset and labelset in parallel
    data = list(zip(imgset, labelset))
    random.shuffle(data)
    imgset, labelset = zip(*data)

    return imgset, labelset

  #torch.utils.dataloaders needs a __getitem__ function to retrieve data
  def __getitem__(self, idx):
    img = Image.open(data_path + str(self.imgsets[idx])).convert('RGB')

    #make sure that the images are 224 by 224
    #transform the image to a tensor and make the pixel values btwn [-1, 1]
    img_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = img_transform(img)

    label = self.labelsets[idx]
    
    return img, label
  
  #torch.utils.dataloaders needs a __len__ function to get the length of the dataset
  def __len__(self):
    return len(self.imgsets)


#generate training, validation, and testing data
batch_size = 64 #? for now, can change later

train_data = datasets('train', data_path)
print(f"The number of training images is: {len(train_data.imgsets)}")
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)

val_data = datasets('val', data_path)
print(f"The number of validation images is: {len(val_data.imgsets)}")
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)

test_data = datasets('test', data_path)
print(f"The number of testing images is: {len(test_data.imgsets)}")
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)

#DATA VISUALIZATION

#making sure dataloaders/associated functions works
#create an iterator for train data
dataiter = iter(train_loader)

#load 1 batch of images and labels
imgs, labels = dataiter.next()

#convert imgs to numpy array for plotting
imgs = imgs.numpy()

#plot images and their labels
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
fig = plt.figure(figsize=(14, 4))
for i in range(8):
  #each image is (2,4) placed at index i+1
  ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
  #transpose the image for plotting
  plt.imshow(np.transpose(imgs[i], (1,2,0)))
  #show the image class aboce the subplot
  ax.set_title(classes[labels[i]])

#CREATING CONVOLUTIONAL MODEL

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()

    self.name = "ConvNet"

    self.num_classes = 9

    #define model layers
    self.conv1 = nn.Conv2d(3, 5, kernel_size=5) #input channels, output channels, kernel, padding
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
    #self.conv3 = nn.Conv2d(10, 20, kernel_size=4)
    self.fc1 = nn.Linear(53*53*10, 32)
    self.classifier = nn.Linear(32, self.num_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    #x = self.pool(F.relu(self.conv3(x)))
    #flatten x for linear layers
    x = x.view(-1, 53*53*10) #each col is 1 input feature, each row is 1 batch example
    x = F.relu(self.fc1(x))
    x = self.classifier(x)
    x = x.squeeze(1) #gets rid of dimensions with size of 1
    return x

#testing if image can be passed through network
# iterator = iter(train_loader)
# images, _ = iterator.next()
# test_model = ConvNet()
# test_img = images[0].unsqueeze(0) #add 1 to first dimension (batch size for test is 1)
# out = test_model(test_img)

#FUNCTION FOR MODEL NAME
def get_model_name(name, batch_size, learning_rate, epoch):
  path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
          batch_size, learning_rate, epoch)
  return path

#CREATING FUNCTION FOR MODEL VALIDATION

def evaluate(model, dataloader, loss_fxn):
  loss = 0.0
  num_correct = 0
  num_examples = 0
  for i, data in enumerate(dataloader, start=0):
    inputs, labels = data
    #forward pass and loss calculation
    outputs = model(inputs)
    loss = loss_fxn(outputs, labels)
    #statistics
    num_examples += len(labels)

    predictions = outputs.max(1, keepdim=True)[1]
    #compute element-wise equality for number of corrects
    #view_as() ensures the labels and predictions tensors are same shape when comparing
    num_correct += predictions.eq(labels.view_as(predictions)).sum().item()

    loss += loss.item()
  
  total_acc = float(num_correct) / num_examples
  total_loss = float(loss) / (i+1)
  
  return total_acc, total_loss

#CREATING TRAIN FUNCTION

def train_net(model, train_loader, val_loader, batch_size=64, learning_rate = 0.01, num_epochs = 30):
  
  #manual seed for consistent results
  torch.manual_seed(1000)

  #define loss function and optimizer
  #use crossentropy loss since this is a classification problem
  loss_fxn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  #set up numpy arrays to store training/validation loss/acc
  train_acc = np.zeros(num_epochs)
  train_loss = np.zeros(num_epochs)
  val_acc = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)

  #start training
  start_time = time.time()
  for epoch in range(num_epochs): #loop over dataset multiple times
    epoch_train_loss = 0.0
    num_correct = 0.0
    num_train_examples = 0

    #i is batch number
    num_iterations = 0
    for inputs, labels in iter(train_loader):
      #forward/backward pass and optimize
      outputs = model(inputs)
      loss = loss_fxn(outputs, labels)
      loss.backward()
      optimizer.step()
      #reset gradients before a new batch is passed through the model
      optimizer.zero_grad()

      #calculate statistics
      #ACCURACY
      #find max along rows (each row corresponds to 1 training ex. with 9 probabilities for each class)
      #keep the orig dimension where num rows = num batchs
      #save the index where the max value occurs -> represents the predicted label
      predictions = outputs.max(1, keepdim=True)[1]
      num_correct += predictions.eq(labels.view_as(predictions)).sum().item()
      num_train_examples += len(labels)
      epoch_train_loss += loss.item()
      num_iterations+=1
    
    train_acc[epoch] = float(num_correct) / num_train_examples #divide by total number of training ex.
    train_loss[epoch] = float(epoch_train_loss) / (num_iterations+1) #divide by number of iterations
    val_acc[epoch], val_loss[epoch] = evaluate(model, val_loader, loss_fxn)

    #print statistics
    print(("Epoch: {} Train Accuracy: {} Train Loss: {} Val Accuracy: {} Val Loss: {}").format(
          epoch+1, train_acc[epoch], train_loss[epoch], val_acc[epoch], val_loss[epoch]))
    
    #save the model for each epoch
    model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
    torch.save(model.state_dict(), model_path)
  
  print("Finished Training")
  end_time = time.time()
  time_passed = end_time - start_time
  print(("Time Elapsed : {:.2f} seconds").format(time_passed))

  #saving statistics for plotting
  np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
  np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
  np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
  np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

def plot_training_curve(path):
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Acc")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

#GENERATING DATASETS FOR SANITY CHECK

class sanity_datasets(Dataset):
  def __init__(self, mode, data_path):
    super(sanity_datasets, self).__init__() 
    self.imgsets, self.labelsets = self.get_data(mode)

  def get_data(self, mode):
    letter_folders = sorted(os.listdir(data_path)) #make sure the letters are in alphabetical order
    label = 0
    #create lists to store images and labels
    imgset = []
    labelset = []
    #for each letter, add the imgs & corresponding label to the lists
    for letter in letter_folders:
      #imgs is a list of all the img names for the current letter
      imgs = os.listdir(data_path + str(letter))
      num_imgs = len(imgs)
      #set the boundaries for training, validation, and testing data
      if mode == 'train':
        start = 0
        stop = int(0.05*num_imgs)
      if mode == 'val':
        start = int(0.05*num_imgs)
        stop = int(0.075*num_imgs)

      #for each image in the current letter folder, append the img name and label to the lists
      for i in range(start, stop):
        imgset.append(str(letter) + '/' + imgs[i]) #ex. A/1_A_1.jpg
        labelset.append(label)
      #increment the label when you move to the next letter
      label +=1

    #shuffle the imgset and labelset in parallel
    data = list(zip(imgset, labelset))
    random.shuffle(data)
    imgset, labelset = zip(*data)

    return imgset, labelset

  #torch.utils.dataloaders needs a __getitem__ function to retrieve data
  def __getitem__(self, idx):
    img = Image.open(data_path + str(self.imgsets[idx])).convert('RGB')

    #transform the image to a tensor and make the pixel values btwn [-1, 1]
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = img_transform(img)

    label = self.labelsets[idx]
    
    return img, label
  
  #torch.utils.dataloaders needs a __len__ function to get the length of the dataset
  def __len__(self):
    return len(self.imgsets)


#generate training, validation, and testing data

sanity_train_data = sanity_datasets('train', data_path)
print(f"The number of sanity training images is: {len(sanity_train_data.imgsets)}")
sanity_train_loader = torch.utils.data.DataLoader(sanity_train_data, batch_size=32, shuffle = True)

sanity_val_data = sanity_datasets('val', data_path)
print(f"The number of sanity validation images is: {len(sanity_val_data.imgsets)}")
sanity_val_loader = torch.utils.data.DataLoader(sanity_val_data, batch_size=32, shuffle = True)

#TRAIN THE NETWORK ON SANITY DATA
sanity_model = ConvNet()
sanity_model.name = "Sanity"
train_net(sanity_model, sanity_train_loader, sanity_val_loader, batch_size=32, learning_rate = 0.05, num_epochs = 30)

#plot the results
sanity_model_path = get_model_name("Sanity", batch_size=32, learning_rate=0.05, epoch=29)
plot_training_curve(sanity_model_path)

#Hyperparameter Search

#FIRST PLOT THE TRAINING CURVE WITH DEFAULT SETTINGS
model = ConvNet()
model.name = "unchanged"
train_net(model, train_loader, val_loader)

unchanged_path = get_model_name("unchanged", batch_size=64, learning_rate=0.01, epoch=29)
plot_training_curve(unchanged_path)

"""Evidently, the model overfits the training data. To fix this I can try reducing batch size and decreasing learning rate."""

#CHANGING BATCH SIZE FROM 64 -> 32, LEARNING RATE 0.01 -> 0.008
train_loader
model = ConvNet()
model.name = "changed_batch"

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)

train_net(model, train_loader, val_loader, batch_size=32, learning_rate=0.008)

changed_batch_path = get_model_name("changed_batch", batch_size=32, learning_rate=0.008, epoch=29)
plot_training_curve(changed_batch_path)

"""The model is still overfitting the training data slightly. I will try increasing the hidden nodes the first linear layer."""

#CHANGING HIDDEN FC NODES 32 -> 64
class new_ConvNet(nn.Module):
  def __init__(self):
    super(new_ConvNet, self).__init__()

    self.name = "ConvNet"

    self.num_classes = 9

    #define model layers
    self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
    self.fc1 = nn.Linear(53*53*10, 64)
    self.classifier = nn.Linear(64, self.num_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    #flatten x for linear layers
    x = x.view(-1, 53*53*10)
    x = F.relu(self.fc1(x))
    x = self.classifier(x)
    x = x.squeeze(1)
    return x

#TRAINING NEW MODEL
model = new_ConvNet()
model.name = "changed_nodes"

batch_size=64
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)

train_net(model, train_loader, val_loader)

changed_nodes_path = get_model_name("changed_nodes", batch_size=64, learning_rate=0.01, epoch=29)
plot_training_curve(changed_nodes_path)

"""The model is still overfitting :( I will try combinations of hyperparameters until I get a better ending validation loss."""

#HYPERPARAM SEARCH, try reducing batch size and learning rate
model = new_ConvNet()
model.name = "try"

batch_size=32
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)

train_net(model, train_loader, val_loader, batch_size=32, learning_rate=0.008)

best_model_path = get_model_name("try", 32, 0.008, 29)
plot_training_curve(best_model_path)

#HYPERPARAM SEARCH
model = new_ConvNet()
model.name = "best"

batch_size=64
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)

train_net(model, train_loader, val_loader, batch_size=64, learning_rate=0.008)

best_model_path = get_model_name("best", 64, 0.008, 29)
plot_training_curve(best_model_path)

#CREATE TEST FUNCTION

def test_net(model, test_loader, batch_size):

  #set up numpy arrays to store training/validation loss/acc
  test_acc = np.zeros(1)

  #start testing
  start_time = time.time()

  test_corrects = 0.0
  num_test_examples = 0

  for inputs, labels in iter(test_loader):
    #forward/backward pass and optimize
    outputs = model(inputs)

    predictions = outputs.max(1, keepdim=True)[1]
    test_corrects += predictions.eq(labels.view_as(predictions)).sum().item()
    num_test_examples += len(labels)
    
  test_acc = float(test_corrects) / num_test_examples #divide by total number of training ex.
  print(("Test Accuracy: {}").format(test_acc))

#loading weights into model
best_model = new_ConvNet()
path = get_model_name("best", 32, 0.008, 29)
state = torch.load(path)
best_model.load_state_dict(state)

#testing the model
test_net(best_model, test_loader, 64)

#IMPLEMENT TRANSFER LEARNING WITH ALEXNET
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

def saveAlexFeatures(data_loader, ftr_path):
  classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
  img_num = 0
  

  for imgs, labels in data_loader:
    img_features = alexnet.features(imgs)
    #stop any weight tracking
    img_features = torch.from_numpy(img_features.detach().numpy())
    #split the batch tensor into individual images
    for i, img_ftr in enumerate(img_features):
      #get rid of the first dimension (batch of 1)
      img_ftr = img_ftr.squeeze(0)
      img_name = classes[labels[i]] + str(img_num)
      path = ftr_path + classes[labels[i]] + '/' + img_name + '.tensor' #name of folder to save img to
      torch.save(img_ftr, path)
      img_num += 1

#SAVING FEATURE EXTRACTED IMAGES

path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/Alex_Features/'

train_ftrs_path = path + 'train_features/'
val_ftrs_path = path + 'val_features/'
test_ftrs_path = path + 'test_features/'

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)

saveAlexFeatures(train_loader, train_ftrs_path)
saveAlexFeatures(val_loader, val_ftrs_path)
saveAlexFeatures(test_loader, test_ftrs_path)

class alexData(Dataset):
  def __init__(self, ftr_path):
    super(alexData, self).__init__() #inheriting functions from the Dataset parent class
    self.ftr_path = ftr_path
    self.imgsets, self.labelsets = self.get_data() #get either train, val, or test data

  def get_data(self):

    letter_folders = sorted(os.listdir(self.ftr_path)) #make sure the letters are in alphabetical order

    label = 0
    #create lists to store images and labels
    imgset = []
    labelset = []
    
    #for each letter, add the imgs & corresponding label to the lists
    for letter in letter_folders:
      #imgs is a list of all the img names for the current letter
      imgs = os.listdir(self.ftr_path + str(letter))
      num_imgs = len(imgs)
      #for each image in the current letter folder, append the img name and label to the lists
      for i in range(num_imgs):
        imgset.append(str(letter) + '/' + imgs[i]) #ex. A/A1.tensor
        labelset.append(label)
      #increment the label when you move to the next letter
      label +=1

    #shuffle the imgset and labelset in parallel
    data = list(zip(imgset, labelset))
    random.shuffle(data)
    imgset, labelset = zip(*data)

    return imgset, labelset

  #torch.utils.dataloaders needs a __getitem__ function to retrieve data
  def __getitem__(self, idx):
    img = torch.load(self.ftr_path + str(self.imgsets[idx]))
    label = self.labelsets[idx]
    
    return img, label
  
  def __len__(self):
    return len(self.imgsets)

#LOAD FEATURE DATA INTO DATALOADERS

path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/Alex_Features/'

train_ftrs_path = path + 'train_features/'
val_ftrs_path = path + 'val_features/'
test_ftrs_path = path + 'test_features/'

batch_size = 64

train_ftr_data = alexData(train_ftrs_path)
train_ftr_loader = torch.utils.data.DataLoader(train_ftr_data, batch_size = batch_size, shuffle = True)

val_ftr_data = alexData(val_ftrs_path)
val_ftr_loader = torch.utils.data.DataLoader(val_ftr_data, batch_size = batch_size, shuffle = True)

test_ftr_data = alexData(test_ftrs_path)
test_ftr_loader = torch.utils.data.DataLoader(test_ftr_data, batch_size = batch_size, shuffle = True)

#Defining new model to process AlexNet features

class alexClassifier(nn.Module):
  def __init__(self):
    super(alexClassifier, self).__init__()

    self.name = "alexClassifier"

    self.num_classes = 9

    #define model layers
    self.fc1 = nn.Linear(256*6*6, 128)
    self.fc2 = nn.Linear(128, 64)
    self.classifier = nn.Linear(64, self.num_classes)

  def forward(self, x):
    #flatten x for linear layers
    x = x.view(-1, 256*6*6)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.classifier(x)
    x = x.squeeze(1)
    return x

#TRAINING THE MODEL
#tracking was removed previously in when features were saved
alex_net = alexClassifier()
train_net(alex_net, train_ftr_loader, val_ftr_loader)

#HYPERPARAMETER SEARCH
train_ftr_loader = torch.utils.data.DataLoader(train_ftr_data, batch_size = 16 , shuffle = True)
val_ftr_loader = torch.utils.data.DataLoader(val_ftr_data, batch_size = 16, shuffle = True)
test_ftr_loader = torch.utils.data.DataLoader(test_ftr_data, batch_size = 16, shuffle = True)

alex_net = alexClassifier()
train_net(alex_net, train_ftr_loader, val_ftr_loader, batch_size=16, learning_rate=0.005, num_epochs=10)

#training curve of best model
best_alex_path = get_model_name("alexClassifier", 16, 0.005, 9)
plot_training_curve(best_alex_path)

best_alexModel = alexClassifier()
path = get_model_name("alexClassifier", 16, 0.005, 9)
state = torch.load(path)
best_alexModel.load_state_dict(state)

#testing the model
test_net(best_alexModel, test_ftr_loader, 64)

"""#Additional Testing With My Own Dataset"""

#DATA PREPROCESSING

#create a dataset class that can be used by torch.utils.dataloader to create batches
class myDatasets(Dataset):
  def __init__(self, path):
    super(myDatasets, self).__init__() #inheriting functions from the Dataset parent class
    self.path = path
    self.imgsets, self.labelsets = self.get_data() #get either train, val, or test data

  def get_data(self):
    letter_folders = sorted(os.listdir(self.path)) #make sure the letters are in alphabetical order
    label = 0
    #create lists to store images and labels
    imgset = []
    labelset = []
    #for each letter, add the imgs & corresponding label to the lists
    for letter in letter_folders:
      #imgs is a list of all the img names for the current letter
      imgs = os.listdir(self.path + str(letter))
      num_imgs = len(imgs)

      #for each image in the current letter folder, append the img name and label to the lists
      for i in range(num_imgs):
        imgset.append(str(letter) + '/' + imgs[i]) #ex. A/1005108786_A_1.jpg
        labelset.append(label)
      #increment the label when you move to the next letter
      label +=1

    #shuffle the imgset and labelset in parallel
    data = list(zip(imgset, labelset))
    random.shuffle(data)
    imgset, labelset = zip(*data)
    return imgset, labelset

  #torch.utils.dataloaders needs a __getitem__ function to retrieve data
  def __getitem__(self, idx):
    img = Image.open(self.path + str(self.imgsets[idx])).convert('RGB')

    #make sure that the images are 224 by 224
    #transform the image to a tensor and make the pixel values btwn [-1, 1]
    img_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = img_transform(img)

    label = self.labelsets[idx]
    
    return img, label
  
  #torch.utils.dataloaders needs a __len__ function to get the length of the dataset
  def __len__(self):
    return len(self.imgsets)


#generate training, validation, and testing data
batch_size = 64 #? for now, can change later

my_data_path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/My_Data/'
my_data = myDatasets(my_data_path)
my_dataloader = torch.utils.data.DataLoader(my_data, batch_size = batch_size, shuffle = True)

#EXTRACTING AND SAVING FEATURES
my_ftr_path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/My_Features/'
saveAlexFeatures(my_dataloader, my_ftr_path)

#This test net will print the number of corrects for each class

def new_test_net(model, test_loader):

  #set up numpy arrays to store training/validation loss/acc
  test_acc = np.zeros(1)

  #start testing
  start_time = time.time()

  test_corrects = 0.0
  num_test_examples = 0

  corrects = [0]*9

  for inputs, labels in iter(test_loader):
    #forward/backward pass and optimize
    outputs = model(inputs)

    predictions = outputs.max(1, keepdim=True)[1]
    if predictions == labels:
      test_corrects+=1
      corrects[labels] +=1

    num_test_examples += len(labels)
    
  test_acc = float(test_corrects) / num_test_examples #divide by total number of training ex.
  print(("Test Accuracy: {}").format(test_acc))
  classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
  for i in range(9):
    print(f"The number of corrects for class {classes[i]} is {corrects[i]}")

#LOADING FEATURE DATA INTO DATALOADER
my_ftr_path = '/content/drive/My Drive/2020-21 School Year/APS360/Lab3/My_Features/'
my_ftr_data = alexData(my_ftr_path)
my_featureloader = torch.utils.data.DataLoader(my_ftr_data, batch_size = 1, shuffle = True)

#testing my model
best_alexModel = alexClassifier()
path = get_model_name("alexClassifier", 16, 0.005, 9)
state = torch.load(path)
best_alexModel.load_state_dict(state)
new_test_net(best_alexModel, my_featureloader)

"""The best accuracy on my test images is 96.3% which is higher than the accuracy for part 4d). This could be due to my images being "good quality" data (ie. hand gestures and features are easily distinguishable from non-relevant image data). My model achieved a perfect accuracy for all letters except 'I', where it predicted 1 image incorrectly. This could be due to that image being more noisy or the positioning of my hand could have looked similar to another letter's sign language representation. As well, only 27 images were tested so each correct prediction causes a great increase in overall accuracy. If more images were tested it is likely that accuracy would not be as high."""
