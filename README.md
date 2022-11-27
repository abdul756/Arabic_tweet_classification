# Arabic-Tweet-Classification-using-transfer-learning-
Automated Arabic Long-Tweet Classification using Transfer Learning with BERT

# TO install Packages run
pip install requirements.txt

# Download dataset  from the given link
https://drive.google.com/drive/folders/18tTQVnkLTKAiVTsnQGFpj58-KY-CCwSH?usp=share_link

# Download model from the given link
https://drive.google.com/file/d/17NSrS0eJPvh0YyEzGSTGQHzGSzdTmSnU/view?usp=share_link

# Description
Social media platforms like Twitter are commonly used by people who are interested in
a wide range of activities and interests. Subjects that may cover their everyday activities and plans, as
well as their thoughts on religion, technology, or products they use. In this paper, we present a BERTbased
text classification model, ARABERT4TWC, for classifying the Arabic tweets of users into different
categories. The purpose of this work is to present a deep learning model to categorize the robust Arabic
tweets of various users in an automated fashion. In our proposed work, a transformer-based model for text
classification is constructed from a pre-trained BERT model provided by the Hugging Face Transformer
library with custom dense layers, and a The classification layer is stacked on top of the BERT encoder to do
the multi-class classification to classify the tweets. First, data sanitation and preprocessing are performed on
the raw Arabic corpus to improve the model’s accuracy. Second, an Arabic-specific BERT model is built,
and input embedding vectors are fed into it. Substantial experiments are run on five publicly available
datasets, and the fine-tuning strategy is evaluated in terms of tokenized vector and learning rate. In addition,
we compare multiple Deep Learning models for Arabic text classification in terms of accuracy.

## How to use

**To Train the model**
``` 
python [-h] [--dataset_name DATASET_NAME]
                             [--dataset_path DATASET_PATH]
                             [--batch_size BATCH_SIZE]
                             [--epoch EPOCH]
                             [--max_token_count MAX_TOKEN_COUNT] 


python --dataset_name which datset to use [for more info check train.py] --dataset_path path_to_your_dataset.csv 

```


**To test the model**
```
model supports  six categories viz politics, religion , finance , tech , sport, medical

python test.py


```

## Results
Task | Metric | hULMonA | AraBERTv0.1 | AraBERTv1 | AraBERTv0.2-base  | AraBERTv0.2-large | CAMeLBERT  | ARABERT4TWC (Our Model)
:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
HARD |Accuracy|0.9570|0.9620|0.9610|-|-|-|0.9749
ASTD |Accuracy|0.771|0.9220|0.9260|0.7690|-|-|0.8610
ArsenTD-Lev|Accuracy|0.5240|0.5356|-|0.5571|0.5694|-|0.5370
AJGT|Accuracy|-|0.9310|0.9380|-|-|-|0.9405
SANAD|Accuracy|-|-|-|-|-|-| 0.9824

