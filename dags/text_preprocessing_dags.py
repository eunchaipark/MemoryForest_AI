from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
import glob
import pandas as pd
import pendulum
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

local_tz = pendulum.timezone("Asia/Seoul")

def clean_html(text):
    return re.sub(r'\s+', ' ', re.sub(r'<.*?>', '', text)).strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def preprocess_blog_json(json_path_pattern):
    files = glob.glob(json_path_pattern)
    results = []
    for file in files:
        with open(file, encoding='utf-8') as f:
            try:
                content = json.load(f)
                for item in content:
                    title = clean_html(item.get('title', ''))
                    description = clean_html(item.get('description', ''))
                    results.append(f"{title} {description}".strip())
            except Exception as e:
                print(f"오류 발생: {file} - {e}")
    return results

def run_sentiment_analysis():
    model_dir = "/opt/airflow/models/saved_kobert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    today_str = datetime.now().strftime("%Y%m%d")
    input_path = f"/opt/airflow/data/blogjson_{today_str}/*_blog.json"
    texts = preprocess_blog_json(input_path)

    results = []
    for text in texts:
        for sent in split_sentences(text):
            pred = classifier(sent)[0]
            label = int(pred['label'])  # 0 또는 1
            score = pred['score']
            results.append({
                'text': sent,
                'label': label,
                'score': score
            })

    df = pd.DataFrame(results)
    output_path = f"/opt/airflow/data/blogjson_{today_str}/sentence_level_sentiment_with_scores.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"분석 결과 저장: {output_path}")

default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='kobert_sentiment_analysis',
    default_args=default_args,
    schedule_interval=None, 
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['nlp', 'sentiment'],
) as dag:
    analyze_task = PythonOperator(
        task_id='run_sentiment_analysis',
        python_callable=run_sentiment_analysis
    )
