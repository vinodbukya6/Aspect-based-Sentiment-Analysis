# -*- coding: utf-8 -*-

"""**Import Modules**"""

# import modules

## Torch, Sklearn imports
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing # label encoder

## PyTorch Transformer
import transformers

## Bert
from transformers import BertModel, BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

## others
import pandas as pd
import numpy as np
import json, re
import random
import uuid
from tqdm import tqdm_notebook

print(torch.__version__)
print(transformers.__version__)

# Install latest Tensorflow build
!pip install -q tf-nightly-2.0-preview

# cuda advantage of the massive parallel computing, 
torch.cuda.is_available() # check cuda is available or not

# set seeding
def random_seeding(seed_value, use_cuda):
  random.seed(seed_value) # pyrhon random
  np.random.seed(seed_value) # numpy
  torch.manual_seed(seed_value) # torch
  if use_cuda: torch.cuda.manual_seed_all(seed_value) # cuda


use_cuda = torch.cuda.is_available()
random_seeding(350, use_cuda)

"""**Uploading csv file**"""

# mount drive to upload data
from google.colab import drive
drive.mount('/content/drive')

# load data
dataset = pd.read_csv('/content/drive/My Drive/aspect_sentiment/absa-cleaned-data.csv')
dataset.head()

dataset.shape

# drop duplicates
dataset = dataset.drop_duplicates(keep=False, inplace=False)
dataset = dataset.reset_index(drop=True)
dataset.shape

# create new datframe for sentiment classification
df = pd.DataFrame(columns = ['text', 'sentiment_label'])
df['text'] = dataset['text']

# extract sentiment from label
# Assumption: overall sentiment for the text if any associated aspect with negative experience is assumed as negative sentiment
for i in range(len(dataset)):
  if 'negative' in dataset.label[i]:
    df['sentiment_label'][i] = 'negative'
  else:
    df['sentiment_label'][i] = 'positive'

df.tail()

"""**Labels text to numbers: Label encoder**"""

# label_encoder object knows how to understand word labels 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels
df['labels']= label_encoder.fit_transform(df['sentiment_label'])

df.head()

df.groupby('labels').size() # 0-negative, 1-positive

"""Model Configurations

Transformers, each model architecture is associated with 3 main types of classes:

**Model class**- to load a particular pre-train model

**Tokenizer class**- to pre-process the data and make it comptible with a particular model 

**Configuration class**- to load the configuration of a particular moel

these classes share a common class method **from_pretrained()**

Example: Bert architecture for text classification

model class - **BertForSequenceClassification**,
tokenizer class- **BertTokenizer**,
configuration class- **BertConfig**
"""

model_type = 'bert' # from HuggingFace
if model_type == 'bert':
    print("BERT")
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = len(list(set(df.labels))) # number of classes in a problem varies
    config.n_layers = 2
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification(config)
    
print(config)

"""**Data pre-processing**

To match pretraining model, we have to format the model input sequence in a specific format, to do we have to tokenize the texts correctly

**Add Special Tokens**

Token [CLS] means the start of a sentence, stands for class [SEP] is for separating sentences for the next sentence prediction task.

The first token of every input sequence is the special classification token – [CLS]. This token is used in classification tasks as an aggregate of the entire sequence representation. It is ignored in non-classification tasks

BERT: [CLS] + tokens + [SEP] + padding
##
DistilBERT: [CLS] + tokens + [SEP] + padding
##
RoBERTa: [CLS] + prefix_space + tokens + [SEP] + padding
##
XLM: [CLS] + tokens + [SEP] + padding
##
XLNet: padding + [CLS] + tokens + [SEP]

**Add special tokens and Zero padding**
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def prepare_features(seq_1, zero_pad = False, max_seq_length = 512):
    enc_text = tokenizer.encode_plus(seq_1, add_special_tokens=True, max_length=300, truncation=True) # add tokens
    if zero_pad:
        while len(enc_text['input_ids']) < max_seq_length:
            enc_text['input_ids'].append(0)
            enc_text['token_type_ids'].append(0)
    return enc_text

temp = []
for i in range(len(df)):
  if len(df['text'][i]) > 512:
    temp.append(i)

len(temp)

len(df['text'][0]), df['text'][0]

"""**Prepare features on texts and labels**"""

class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        utterance = self.data.text[index]
        X = prepare_features(utterance, zero_pad = True)
        y =  int(self.data.labels[index])
        return np.array(X['input_ids']), np.array(X['token_type_ids']), np.array(y)
    
    def __len__(self):
        return self.len

"""Splitting the data"""

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

df.index, train_dataset.index, test_dataset.index

df.shape, train_dataset.shape, test_dataset.shape

train_dataset.head()

training_set = Intents(train_dataset)
testing_set = Intents(test_dataset)

"""**Dataloaders and Parameters**"""

### Dataloaders Parameters
params = {'batch_size': 16,
          'shuffle': True,
          'drop_last': True,
          'num_workers': 0}
training_loader = DataLoader(training_set, **params)
testing_loader = DataLoader(testing_set, **params)
loss_function = nn.CrossEntropyLoss()
learning_rate = 2e-05 
optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    print("GPU is AVAILABLE!")
    model = model.cuda()

ids, tokens, labels = next(iter(training_loader)) # iterated one element at a time
ids.shape, tokens.shape, labels

if model_type == 'bert':
    print(model_type)
    out = model.forward(ids.cuda())[0]

print(loss_function(out, labels.cuda()))
print(out.shape)

"""**Training the model**"""

def train(model, epochs):
  max_epochs = epochs
  model = model.train()
  for epoch in tqdm_notebook(range(max_epochs)):
      print("EPOCH -- {}".format(epoch))
      correct = 0
      total = 0
      for i, (ids, tokens, labels) in enumerate(training_loader):
          optimizer.zero_grad()
          if torch.cuda.is_available():
              ids = ids.cuda()
              tokens = tokens.cuda()
              labels = labels.cuda()
      
          if model_type == 'bert':
              output = model.forward(ids)[0]

          loss = loss_function(output, labels)
          loss.backward()
          optimizer.step()

          _, predicted = torch.max(output.data, 1)
          total += labels.size(0)
          correct += (predicted.cpu() == labels.cpu()).sum()
      train_accuracy = 100.00 * correct.numpy() / total
      print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), train_accuracy))
  return "Training Finished"

train(model, 20) # 20 epochs

def evaluate_accuracy(model):
  correct = 0
  total = 0
  num_classes = 3
  # Initialize the prediction and label lists(tensors)
  pred_list = torch.zeros(0,dtype=torch.long)
  label_list = torch.zeros(0,dtype=torch.long)
  for (ids, tokens, labels) in testing_loader:
      if torch.cuda.is_available():
          ids = ids.cuda()
          tokens = tokens.cuda()
          labels = labels.cuda()

      if model_type == 'bert':
          output = model.forward(ids)[0]

      _, predicted = torch.max(output.data, 1)

      # Append batch prediction results
      # torch.cat- concatenates along an existing dimension. and so the number of dimensions of the output is the same as the inputs.
      pred_list = torch.cat([pred_list, predicted.view(-1).cpu()])
      label_list = torch.cat([label_list, labels.view(-1).cpu()])

     

      total += labels.size(0)
      correct += (predicted.cpu() == labels.cpu()).sum()
  # Accuracy
  accuracy = 100.00 * correct.numpy() / total

  # Confusion matrix
  conf_matrix = confusion_matrix(label_list.numpy(), pred_list.numpy())

  # Classification_report
  classify_report = classification_report(label_list.numpy(), pred_list.numpy())
  print('classification report:', classify_report)

  return accuracy, conf_matrix

evaluate_accuracy(model) # seed-350


label_to_ix = {'neg': 0, 'pos': 1}

label_to_ix.keys(), label_to_ix.values()

"""**Predictions**"""

def predict(text, language = 'en'):
      model.eval()
      features = prepare_features(text, zero_pad = True)
      ids = torch.tensor(features['input_ids']).unsqueeze(0)
      tokens = torch.tensor(features['token_type_ids']).unsqueeze(0)
      if torch.cuda.is_available():
          ids = ids.cuda()
          tokens = tokens.cuda()
      if model_type == 'bert':
        logits_out = model.forward(ids)[0].squeeze(0)

      softmax_out = F.softmax(logits_out, dim=0)
      _, pred_label = torch.max(softmax_out.data, 0)

      prediction=list(label_to_ix.keys())[list(label_to_ix.values()).index(pred_label.data.cpu())]
 
      return prediction

# predictions
sent1 = 'service is good and price is cheap but quality is poor'
predict(sent1)
# output 'negative'
