
# installation
#!pip install simpletransformers

# install iterative stratification for labelsets
#!pip install iterative-stratification

# import libraries
## torch and simpletransformers
import torch
import transformers
## simpletransformers
from simpletransformers.classification import ClassificationModel
## sklearn
from sklearn import preprocessing # label encoder
## others
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm_notebook

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

dataset.shape

# create new datframe for sentiment classification
df = pd.DataFrame(columns = ['text', 'sentiment_label'])
df['text'] = dataset['text']

# extract sentiments
# Assumption: overall sentiment for the text if any associated labels with negative experience is assumed as negative sentiment
for i in range(len(dataset)):
  if 'negative' in dataset.label[i]:
    df['sentiment_label'][i] = 'negative'
  else:
    df['sentiment_label'][i] = 'positive'

df.head()

# label_encoder object knows how to understand word labels 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels
df['labels']= label_encoder.fit_transform(df['sentiment_label'])

df.head()

# drop sentimnet_label column, because model takes only text, labels(one hot encoding) as inputs
df1 = df.drop(['sentiment_label'], axis=1)
df1.head()

df1.labels.value_counts() # number samples per label

"""**Split dataset into train, test, eval sets**"""

# split the data into train and test
train_size = 0.9
train_dataset = df1.sample(frac=train_size, random_state=50).reset_index(drop=True)
test_dataset1 = df1.drop(train_dataset.index).reset_index(drop=True)

train_dataset.labels.value_counts(), test_dataset1.labels.value_counts()

# split the test_dataset1 into eval and test sets
train_size = 0.65
eval_dataset = test_dataset1.sample(frac=train_size, random_state=50).reset_index(drop=True)
test_dataset = test_dataset1.drop(eval_dataset.index).reset_index(drop=True)

train_dataset.labels.value_counts(), eval_dataset.labels.value_counts(), test_dataset.labels.value_counts()

train_dataset.shape, eval_dataset.shape, test_dataset.shape


# create model 
model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=2,
                                      args={'fp16': False, 'overwrite_output_dir': True, 'do_lower_case': True,
                                            'evaluate_during_training': True, 'evaluate_during_training_steps': 400, 'save_steps': 1000,
                                            'learning_rate': 2e-5, 'num_train_epochs': 10, 'logging_steps': 50, 'adam_epsilon': 1e-6, 'evaluate_during_training_verbose':True,
                                            'max_seq_length':512, 'reprocess_input_data': True, 'tensorboard_folder': None,
                                            'train_batch_size': 16, 'eval_batch_size': 12, 'weight_decay': 0.01, 'manual_seed': 9,
                                            'use_early_stopping': True, 'early_stopping_patience': 3, 'early_stopping_delta': 0.001})

# for small dataset: train more epochs, optimal batch size 8 to 16
# DistilBert- lr:5e-06, batch_size:8, epochs:20
# for our data- lr: 2e-5, adam_epsilon: 1e-6, weight_decay: 0.01
# logging_steps: log training loss and learning at every specified number of steps

# Train the model
model.train_model(train_df=train_dataset, eval_df=eval_dataset)

scores = pd.read_csv('outputs/training_progress_scores.csv')
scores

# while using predict() function should not use cache 
# always check 'Cache is not used'
sentiment_checkpoint = ClassificationModel('distilbert', 'outputs/checkpoint-2800/', num_labels=2, use_cuda='cuda',
                               args={'use_cached_eval_features': False, 'reprocess_input_data':True,
                                     'silent': True, 'do_lower_case': True})

# evalution set
result, model_outputs, wrong_predictions = sentiment_checkpoint.eval_model(eval_dataset, multi_label=False,verbose=True)

result

to_predict = test_dataset.text.tolist()
preds, outputs = sentiment_checkpoint.predict(to_predict)

y_true = list(test_dataset['labels'])
y_true = np.array(y_true)
predictions_test = np.array(preds)

from sklearn import metrics
print(metrics.classification_report(y_true, predictions_test))

true_transformed = label_encoder.inverse_transform(y_true) # true labels in test set
pred_transformed = label_encoder.inverse_transform(predictions_test) # predicted labels of test set

for (i, text, t, p) in zip(test_dataset.index, test_dataset.text, true_transformed, pred_transformed):
    print(i, f'text_raw: {text} \n'
          f'true labels:  {t} \n'
          f'predicted as: {p}\n')

text = ['service is good but late fitting']
preds, outputs = sentiment_checkpoint.predict(text)
preds

