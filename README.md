# Aspect-based Sentiment Analysis
Aspect-based sentiment analysis goes one step further than sentiment analysis by automatically assigning sentiments to specific aspects or topics. It involves breaking down text data into smaller fragments, allowing you to obtain more granular and accurate insights from data.
Approach: For a single text, pair it with all the aspects. Let the model predict whether the aspect is present and if yes then what’s the sentiment.

# 1. Aspect classifier
The learning of aspects is a multi-label classification task. Total 30 labels.
# 2. Sentiment classifier 
This is a single-label classification task. Given a text and the recognised aspects in it, we have to predict whether the text has a negative or positive sentiment with respect to the aspect. Assumption Overall sentiment of the text is negative if any associated aspects with negative experience.

# Models used
Pretrained models like BERT, RoBERTa, XLNet, DistilBERT, etc (from HuggingFace transformers) boosts performance in many small-data tasks. I have used a simple transformers library(https://github.com/ThilinaRajapakse/simpletransformers). This library is based on the Transformer library by HuggingFace. To quickly train and evaluate Transformer models. For Unsupervised Data Augmentation nlpaug library is used.

# Next Steps for improving the model performance
Label more data with pre-selected important aspects. For better performance we need at least 100 samples per aspect(model giving good performance for more than 100 samples per aspect).
Model selection: I have used DistilBERT for fast training procedure, but for best prediction metrics use Facebook’s RoBERTa, BERT, XLNet’s.
Use BERT or RoBERTa (or any other language model)+ UDA with different data augmentation techniques
