import torch

import torch.nn as nn





from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
import torch
from tweet_classification import TweetClassification




file = input("Enter your tweet ")



BERT_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

LABEL_COLUMNS=['category is_Finance',
 'category is_Medical',
 'category is_Politics',
 'category is_Religion',
 'category is_Sports',
 'category is_Tech']

trained_model = TweetClassification(


  n_classes=len(LABEL_COLUMNS)

)



trained_model.load_state_dict(torch.load('model_arabic.pt'))
trained_model.eval()
trained_model.freeze()



def tweet_classify():

    THRESHOLD=0.8

    # file = "ارتفع مؤشر سوق الإمارات المالي الصادر عن هيئة الأوراق المالية والسلع خلال جلسة أمس بنسبة 44 .1% ليغلق على"
    encoding = tokenizer.encode_plus(

    file,

    add_special_tokens=True,

    max_length=512,

    return_token_type_ids=False,

    padding="max_length",

    return_attention_mask=True,

    return_tensors='pt',

    )

    _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])

    test_prediction = test_prediction.flatten().numpy()

    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        if prediction < THRESHOLD:
            continue

        print(f"{label} : {prediction}")
            
tweet_classify()