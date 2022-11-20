

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import pandas as pd
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
import pytorch_lightning as pl

from torchmetrics.functional import accuracy, f1, auroc

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process(final_data):
  df1 = pd.get_dummies(final_data.label, prefix = "Tweet_Category_is_")
  final_data=pd.concat([final_data, df1], axis=1)
  return final_data



def main(args):

  if args.dataset_name == "SANAD":
    df = pd.read_csv(args.dataset_path)
    df.rename(columns = {'text':'Tweet'}, inplace = True)
    final_data = df


  elif args.dataset_name == "HARD":
    df = pd.read_csv(args.dataset_path)
    df.rename(columns = {'text':'Tweet', 'labels':'Label_encoding'}, inplace = True)
    dict = {0:'negative',  1:'positive'}
    df['label'] = df['Label_encoding'].map(dict)
    final_data = df
    final_data=process(final_data)

  elif args.dataset_name == "ASTD":
    df = pd.read_csv(args.dataset_path)
    df.rename(columns = {'text':'Tweet', 'label':'Label_encoding'}, inplace = True)
    dict = {0:'NEG', 1:'POS', 2:'NEUTRAL', 3:'OBJ'}
    df['label'] = df['Label_encoding'].map(dict)
    final_data = df
    final_data=process(final_data)

  elif args.dataset_name == "ARSENTD":
    df = pd.read_csv(args.dataset_path)
    dict = {'negative':0, 'neutral':1, 'positive':2, 'very_negative':3, 'very_positive':4}
    df['Label_Encoding'] = df['label'].map(dict)
    final_data = df
    final_data=process(final_data)

  else:
    df = pd.read_csv(args.dataset_path)
    final_data = df




  print("loading {} dataset".format(args.dataset_name))
  train_df, val_df = train_test_split(final_data, test_size=0.2)

  train_df.shape, val_df.shape

  LABEL_COLUMNS = final_data.columns.tolist()[3:]
  LABEL_COLUMNS
  print(LABEL_COLUMNS)

  MAX_TOKEN_COUNT = args.max_token_count
  BERT_MODEL_NAME = 'bert-base-cased'

  tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

  class Dataset_(Dataset):
    def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: BertTokenizer,
      max_token_len: int = 128
    ):
      self.tokenizer = tokenizer
      self.data = data
      self.max_token_len = max_token_len
    def __len__(self):
      return len(self.data)
    def __getitem__(self, index: int):
      data_row = self.data.iloc[index]
      comment_text = data_row.Tweet
      labels = data_row[LABEL_COLUMNS]
      encoding = self.tokenizer.encode_plus(
        comment_text,
        add_special_tokens=True,
        max_length=self.max_token_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
      )
      return {'comment_text':comment_text,
        'input_ids':encoding["input_ids"].flatten(),
        'attention_mask':encoding["attention_mask"].flatten(),
        'labels':torch.FloatTensor(labels)}

  class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
      super().__init__()
      self.batch_size = batch_size
      self.train_df = train_df
      self.test_df = test_df
      self.tokenizer = tokenizer
      self.max_token_len = max_token_len
    def setup(self, stage=None):
      self.train_dataset = Dataset_(
        self.train_df,
        self.tokenizer,
        self.max_token_len
      )
      self.test_dataset = Dataset_(
        self.test_df,
        self.tokenizer,
        self.max_token_len
      )
    def train_dataloader(self):
      return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=4
      )
    def val_dataloader(self):
      return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=2
      )
    def test_dataloader(self):
      return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=4
      )

  N_EPOCHS =args.epoch
  BATCH_SIZE = args.batch_size
  data_module = DataModule(
    train_df,
    val_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
  )

  class TweetClassification(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
      super().__init__()
      self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
      self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
      self.n_training_steps = n_training_steps
      self.n_warmup_steps = n_warmup_steps
      self.criterion = nn.BCELoss()
    def forward(self, input_ids, attention_mask, labels=None):
      output = self.bert(input_ids, attention_mask=attention_mask)
      output = self.classifier(output.pooler_output)
      output = torch.sigmoid(output)
      loss = 0
      if labels is not None:
          loss = self.criterion(output, labels)
      return loss, output
    def training_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]
      labels = batch["labels"]
      loss, outputs = self(input_ids, attention_mask, labels)
      self.log("train_loss", loss, prog_bar=True, logger=True)
      return {"loss": loss, "predictions": outputs, "labels": labels}
    def validation_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]
      labels = batch["labels"]
      loss, outputs = self(input_ids, attention_mask, labels)
      self.log("val_loss", loss, prog_bar=True, logger=True)
      return loss
    def test_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]
      labels = batch["labels"]
      loss, outputs = self(input_ids, attention_mask, labels)
      self.log("test_loss", loss, prog_bar=True, logger=True)
      return loss
    def training_epoch_end(self, outputs):
      labels = []
      predictions = []
      for output in outputs:
        for out_labels in output["labels"].detach().cpu():
          labels.append(out_labels)
        for out_predictions in output["predictions"].detach().cpu():
          predictions.append(out_predictions)
      labels = torch.stack(labels).int()
      predictions = torch.stack(predictions)
      for i, name in enumerate(LABEL_COLUMNS):
        class_roc_auc = auroc(predictions[:, i], labels[:, i])
        self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    def configure_optimizers(self):
      optimizer = AdamW(self.parameters(), lr=4e-5)
      scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.n_warmup_steps,
        num_training_steps=self.n_training_steps
      )
      return {'optimizer': optimizer, 'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}



  val_dataset = Dataset_(
    val_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
  )

  from tqdm import tqdm




  trained_model =TweetClassification.load_from_checkpoint("logs/tweet-comments/model.ckpt",
    n_classes=len(LABEL_COLUMNS)
  )

  trained_model = trained_model.to(device)
  trained_model.eval()
  trained_model.freeze()





  predictions = []
  labels = []
  for item in tqdm(val_dataset):
    _, prediction = trained_model(
      item["input_ids"].unsqueeze(dim=0).to(device),
      item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    predictions.append(prediction.flatten())
    labels.append(item["labels"].int())
  predictions = torch.stack(predictions).detach().cpu()
  labels = torch.stack(labels).detach().cpu()

  THRESHOLD = 0.5

  y_pred = predictions.numpy()
  y_true = labels.numpy()
  upper, lower = 1, 0
  y_pred = np.where(y_pred > THRESHOLD, upper, lower)
  report_dict = classification_report(
    y_true,
    y_pred,
    target_names=LABEL_COLUMNS,
    zero_division=0
  )



  data  = pd.DataFrame(report_dict).transpose()
  data.to_csv("/content/gdrive/MyDrive/Arabic_distillation/lightning_logs_128/classification_report_128.csv", index=True)


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--dataset_name", type=str, default="SANAD", help= "[SANAD, HARD, ARSENTD,ASTD, AJGT]")
    parser.add_argument("--dataset_path", type=str, default="/content/gdrive/MyDrive/Dataset/sanad.csv", help="The directory to the dataset file")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for the generator")
    parser.add_argument("--epoch", type=int, default=2, help="number of the samples to infer.")
    parser.add_argument("--max_token_count", type=int, default=128, help="Maxium number of tokens in the input.")


    args = parser.parse_args()
    main(args)