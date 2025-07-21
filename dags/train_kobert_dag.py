# dags/train_kobert_dag.py

import os
import csv
import shutil
import torch
import pendulum
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from airflow import DAG
from airflow.operators.python import PythonOperator

local_tz = pendulum.timezone("Asia/Seoul")

F1_WEIGHT = 0.6
RECALL_WEIGHT = 0.4

def train_model(folder_date: str = None):
    if folder_date is None:
        folder_date = datetime.now().strftime("%Y%m%d")

    csv_path = f'/opt/airflow/data/sentiment_results/{folder_date}/sentence_level_sentiment_with_scores.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path).sample(frac=0.5, random_state=42)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    saved_model_path = "/opt/airflow/models/saved_kobert_model"
    backup_model_path = "/opt/airflow/models/saved_kobert_model_backup"
    temp_model_path = "/opt/airflow/models/saved_kobert_model_temp"

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path, use_fast=True)

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

    # 1️⃣ 기존 모델 불러오기 (가중치 유지)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path, num_labels=2)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    training_args = TrainingArguments(
        output_dir=temp_model_path,
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

    # 2️⃣ 점수 계산
    accuracy = results.get("eval_accuracy", 0)
    f1 = results.get("eval_f1", 0)
    precision = results.get("eval_precision", 0)
    recall = results.get("eval_recall", 0)
    new_score = (f1 * F1_WEIGHT) + (recall * RECALL_WEIGHT)

    # 3️⃣ 이전 점수 불러오기
    metric_path = '/opt/airflow/data/metrics.csv'
    prev_score = 0.0
    if os.path.exists(metric_path):
        with open(metric_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(',')
                try:
                    prev_f1 = float(last[3])
                    prev_recall = float(last[5])
                    prev_score = (prev_f1 * F1_WEIGHT) + (prev_recall * RECALL_WEIGHT)
                except:
                    pass

    # 4️⃣ 성능 비교 → 저장 여부 결정
    if new_score >= prev_score:
        print("[✔] 성능 향상 → 모델 덮어쓰기 및 백업")
        model.save_pretrained(saved_model_path)
        tokenizer.save_pretrained(saved_model_path)

        if os.path.exists(backup_model_path):
            shutil.rmtree(backup_model_path)
        shutil.copytree(saved_model_path, backup_model_path)

    else:
        print("[✘] 성능 저하 → 기존 모델 유지")

    # 5️⃣ 성능 기록
    now = datetime.now()
    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        round(accuracy, 4),
        round(f1, 4),
        round(precision, 4),
        round(recall, 4),
    ]

    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    file_exists = os.path.isfile(metric_path)
    with open(metric_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['date', 'time', 'accuracy', 'f1', 'precision', 'recall'])
        writer.writerow(row)


# ----------------------
# DAG 설정
# ----------------------

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='kobert_incremental_training_with_backup',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['ml', 'kobert']
) as dag:

    train_kobert_task = PythonOperator(
        task_id='train_kobert_incrementally',
        python_callable=train_model,
        op_kwargs={'folder_date': datetime.now().strftime('%Y%m%d')}
    )
