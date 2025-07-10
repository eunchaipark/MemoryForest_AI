from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pendulum
import os
import json
import glob
import pandas as pd
import re
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Timezone
local_tz = pendulum.timezone("Asia/Seoul")

# Constants
MODEL_PATH = "/opt/airflow/models/saved_kobert_model"
DATA_BASE_PATH = "/opt/airflow/data"

# Text Preprocessing
def clean_html(text):
    return re.sub(r'\s+', ' ', re.sub(r'<.*?>', '', text)).strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def preprocess_blog_json(json_path_pattern):
    # glob 패턴을 사용해 모든 주제/날짜 하위 json 파일들을 수집
    files = glob.glob(json_path_pattern, recursive=True)  # recursive=True 권장
    results = []
    for file in files:
        try:
            with open(file, encoding='utf-8') as f:
                content = json.load(f)
                for item in content:
                    title = clean_html(item.get('title', ''))
                    description = clean_html(item.get('description', ''))
                    results.append(f"{title} {description}".strip())
        except Exception as e:
            logging.warning(f"[전처리 오류] {file} - {e}")
    return results

def run_sentiment_analysis(folder_date: str = None):
    if folder_date is None:
        folder_date = datetime.now().strftime("%Y%m%d")

    # 수정된 경로: 주제 폴더 하위 날짜별 json 파일들을 모두 읽음
    input_path = os.path.join(DATA_BASE_PATH, "*", folder_date, "*_blog.json")
    texts = preprocess_blog_json(input_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    results = []
    for text in texts:
        for sent in split_sentences(text):
            try:
                pred = classifier(sent)[0]
                results.append({
                    'text': sent,
                    'label': int(pred['label']),  # 0 or 1
                    'score': pred['score']
                })
            except Exception as e:
                logging.error(f"[분석 오류] 문장: {sent} - {e}")

    # 결과는 날짜별 단일 폴더에 통합 저장
    output_dir = os.path.join(DATA_BASE_PATH, "sentiment_results", folder_date)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sentence_level_sentiment_with_scores.csv")
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info(f"[분석 완료] 결과 저장: {output_path}")


# DAG 정의
default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='kobert_sentiment_analysis_v2',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['nlp', 'sentiment', 'kobert'],
) as dag:

    analyze_task = PythonOperator(
        task_id='run_kobert_sentiment_analysis',
        python_callable=run_sentiment_analysis,
        op_kwargs={'folder_date': datetime.now().strftime('%Y%m%d')}  # 필요시 수동 지정 가능
    )
