
# import libraries
## torch and simpletransformers
import torch
import transformers
import simpletransformers
from simpletransformers.classification import MultiLabelClassificationModel
## sklearn
# MultilabelStratifiedShuffleSplit is used for splitting train,test,eval sets. atleast one sample present per aspect in each set.
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit 
from sklearn.preprocessing import MultiLabelBinarizer # one hot encoding for multilabels
from sklearn import metrics
from sklearn.utils import shuffle

## augmentation required libraries
import nlpaug.augmenter.word as naw
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

## others
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm

# check versions
print(torch.__version__)
print(transformers.__version__)

torch.cuda.is_available()

# set seeding
def random_seeding(seed_value, use_cuda):
  random.seed(seed_value) # pyrhon random
  np.random.seed(seed_value) # numpy
  torch.manual_seed(seed_value) # torch
  if use_cuda: torch.cuda.manual_seed_all(seed_value) # cuda


use_cuda = torch.cuda.is_available()
random_seeding(9, use_cuda)

# mount drive to upload data
from google.colab import drive
drive.mount('/content/drive')

# load data
dataset = pd.read_csv('/content/drive/My Drive/aspect_sentiment/absa-cleaned-data.csv')
dataset.head()

# check shape
dataset.shape

# drop duplicates
dataset = dataset.drop_duplicates(keep=False, inplace=False)
dataset.shape #

# 'list_labs' column in string convert into list
x = dataset['list_aspects'].apply(eval)
len(x), x[220], type(x[0])

# Create MultiLabelBinarizer object
mlb = MultiLabelBinarizer()

# One-hot encode data
one_hot_en = mlb.fit_transform(x)

# list of all labels
all_labels = mlb.classes_ 
len(one_hot_en), len(all_labels) # total 30 labels

# one_hot_en to dataframe
dataset['labels'] = list(one_hot_en)

dataset.tail()

# number of examples per label
for j in range(len(all_labels)):
    count = 0
    for x in dataset['labels']:
        if x[j] ==1:
            count +=1
    print(j, all_labels[j], count)

# data for splitting
X = dataset['text'].values.astype('U')
y = dataset['labels'].values

# MultilabelStratifiedShuffleSplit takes lists
text1 = dataset['text'].tolist()
label1 = dataset['labels'].tolist()

# creating shuffle split
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9)

# splitting
for train_index, test_index in msss.split(text1, label1):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train1, X_test = X[train_index], X[test_index]
    y_train1, y_test = y[train_index], y[test_index]

# checking number of examples per label in evaluation set,
for j in range(len(all_labels)):
    count = 0
    for x in y_test:
        if x[j] ==1:
            count +=1
    if count == 0:
      print(j, all_labels[j], count) #

# evaluation dataframe
eval_df = pd.DataFrame(list(zip(X_test, y_test)), columns=['text', 'labels'])
eval_df.head()

# train test dataframe for splitting training and testing
train_test_df = pd.DataFrame(list(zip(X_train1, y_train1)), columns=['text', 'labels'])
train_test_df.head()

# data for splitting
X = train_test_df['text'].values.astype('U')
y = train_test_df['labels'].values

# MultilabelStratifiedShuffleSplit takes lists 
text2 = train_test_df['text'].tolist()
label2 = train_test_df['labels'].tolist()

# splitting
for train_index, test_index in msss.split(text2, label2):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]

# testing dataframe
test_df = pd.DataFrame(list(zip(X_eval, y_eval)), columns=['text', 'labels'])
test_df.head()

# number of samples per label in train set
for j in range(len(all_labels)):
    count = 0
    for x in y_train:
        if x[j] ==1:
            count +=1
    print(j, all_labels[j], count)

# training dataframe 
train_df1 = pd.DataFrame(list(zip(X_train, y_train)), columns=['text', 'labels'])
train_df1.head()

# all_labels indexes number samples per label < 100
less_samples = [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 20, 21, 22, 23, 24, 25, 26]
len(less_samples)

"""**Data Augmentation**

Using 'nlpaug' library
"""

# substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
aug_context = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
# substitute word by WordNet's synonym after contextual augmentation
aug_syn = naw.SynonymAug(aug_src='wordnet')
#augmented_text = aug.augment(text, n=3)

less_samples, all_labels

## augmentation
temp_text = [] # for augmented text
temp_label = [] # for labels
# looping for less_samples
for x in tqdm(less_samples):
  #x = less_samples[i]
  for j in range(len(train_df1)):
    # store label 
    label = train_df1.labels[j]
    if label[x] == 1:
      # store text
      text = train_df1.text[j]
      aug_1 = aug_context.augment(text, n=3, num_thread=5) # contextual word embedding    
      for k in range(len(aug_1)):
        # WordNets synonym after contextual word embedding
        temp_text.append(aug_syn.augment(aug_1[k], num_thread=3))
        temp_label.append(label)

len(temp_text), len(temp_label)

temp_text[0:5], temp_label[0:5]

# augmented dataframe 
aug_df1 = pd.DataFrame(list(zip(temp_text, temp_label)), columns=['text', 'labels'])
aug_df1.head()

training_df = shuffle(train_df1.append(aug_df1, ignore_index=True))
training_df.shape

# number of samples per label
for j in range(len(all_labels)):
    count = 0
    for x in y_train:
        if x[j] ==1:
            count +=1
    print(j, all_labels[j], count)

# reset index
training_df = training_df.reset_index(drop=True)
training_df.head(10)

training_df.shape, eval_df.shape, test_df.shape # shapes


# create model 
model = MultiLabelClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=30,
                                      args={'fp16': False, 'overwrite_output_dir': True, 'do_lower_case': True,
                                            'evaluate_during_training': True, 'evaluate_during_training_steps': 500, 'save_steps': 1500,
                                            'learning_rate': 2e-5, 'num_train_epochs': 15, 'logging_steps': 50, 'adam_epsilon': 1e-6, 'evaluate_during_training_verbose':True,
                                            'max_seq_length':512, 'reprocess_input_data': True, 'tensorboard_folder': None,
                                            'train_batch_size': 12, 'eval_batch_size': 8, 'weight_decay': 0.01, 'manual_seed': 9,
                                            'use_early_stopping': True, 'early_stopping_patience': 5, 'early_stopping_delta': 0.001})

# for small dataset: train more epochs, optimal batch size 8 to 16
# DistilBert- lr:5e-06, batch_size:8, epochs:20
# for our data- lr: 2e-5, adam_epsilon: 1e-6, weight_decay: 0.1
# logging_steps: log training loss and learning at every specified number of steps

# Train the model
model.train_model(train_df=training_df, eval_df=eval_df)

scores = pd.read_csv('outputs/training_progress_scores.csv')
scores

# predict() function should not use cache 
# always check 'Cache is not used'
aspect_checkpoint = MultiLabelClassificationModel('distilbert', 'outputs/checkpoint-5000/', num_labels=30, use_cuda='cuda', 
                                                  args={'use_cached_eval_features': False, 'reprocess_input_data':True,
                                                        'silent': True, 'do_lower_case': True})

# results
result, model_outputs, wrong_predictions = aspect_checkpoint.eval_model(eval_df, multi_label=True,verbose=True)

result

# model takes list of text
#example: model.predict(['Some arbitary sentence'])
to_predict = test_df.text.tolist()
preds, outputs = aspect_checkpoint.predict(to_predict)

# transforming data for classification report
y_true = list(test_df['labels'])
y_true = np.array(y_true)
predictions_test = np.array(preds)

print(metrics.classification_report(y_true, predictions_test))

print(metrics.classification_report(y_true, predictions_test))

labels_df = pd.DataFrame(all_labels, columns=['aspect_labels'])
# to save
labels_df.to_csv('/content/drive/My Drive/aspect_sentiment/all_aspect_labels.csv', index=False)

all_labels

# predictions
text = ['it was so easy to book and arrange local fitting. the prices were competitive']
preds, outputs = aspect_checkpoint.predict(text)
for i in range(len(all_labels)):
  if preds[0][i] == 1:
    print(all_labels[i])