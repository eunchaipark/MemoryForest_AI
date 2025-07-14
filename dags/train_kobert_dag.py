# dags/train_kobert_dag.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime, timedelta
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import csv

local_tz = pendulum.timezone("Asia/Seoul")

def train_model(folder_date: str = None):
    if folder_date is None:
        folder_date = datetime.now().strftime("%Y%m%d")

    csv_path = f'/opt/airflow/data/sentiment_results/{folder_date}/sentence_level_sentiment_with_scores.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    # 데이터 샘플링 (메모리 절약)
    df = df.sample(frac=0.4, random_state=42)

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    model_dir = "/opt/airflow/models/saved_kobert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    training_args = TrainingArguments(
        output_dir='/opt/airflow/models/kobert_finetuned',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir='/opt/airflow/logs',
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()

    metrics_file = '/opt/airflow/data/metrics.csv'
    now = datetime.now()
    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        results.get('eval_accuracy', ''),
        results.get('eval_f1', ''),
        results.get('eval_precision', ''),
        results.get('eval_recall', '')
    ]

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['date', 'time', 'accuracy', 'f1', 'precision', 'recall'])
        writer.writerow(row)

# --------------------------
# Airflow DAG 설정
# --------------------------

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='kobert_finetune_dag',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['ml', 'kobert']
) as dag:

    train_kobert_task = PythonOperator(
        task_id='train_kobert_model',
        python_callable=train_model,
        op_kwargs={'folder_date': datetime.now().strftime('%Y%m%d')}
    )