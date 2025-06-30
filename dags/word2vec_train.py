from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv
import time
import os
import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Okt
import pendulum

local_tz = pendulum.timezone("Asia/Seoul")

def train_word2vec():
    start_time = time.time()
    # 날짜 기준 파일 경로 설정
    today_str = datetime.now().strftime("%Y%m%d")
    csv_path = f"/opt/airflow/data/blogjson_{today_str}/sentence_level_sentiment_with_scores.csv"
    # csv_path = f"/opt/airflow/data/blogjson_20250626/sentence_level_sentiment_with_scores.csv"
    model_save_path = "/opt/airflow/models/word2vec_custom.model"  # 덮어쓰기 경로
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df['text'].dropna().drop_duplicates()

    okt = Okt()
    def tokenize_text(text):
        allowed_pos = ['Noun', 'Verb', 'Adjective']
        return [word for word, pos in okt.pos(text, stem=True) if pos in allowed_pos]

    tokenized_sentences = [tokenize_text(line.strip()) for line in texts if line.strip()]

    print("Word2Vec 모델 학습 시작...")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1
    )
    model.save(model_save_path)
    print(f"Word2Vec 모델 저장 완료: {model_save_path}")

    # 모델 학습 및 저장 후 추가
    train_stats_path = "/opt/airflow/data/word2vec_train_stats.csv"
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_time = datetime.now().strftime("%H:%M:%S")

    # 모델 정보 추출
    train_time_sec = round(time.time() - start_time, 2)
    vocab_size = len(model.wv)
    vector_size = model.vector_size
    raw_word_count = model.corpus_total_words
    effective_word_count = model.corpus_count
    words_per_sec = int(effective_word_count / train_time_sec)
    alpha = model.alpha


    # 파일이 없다면 헤더 먼저 쓰기
    file_exists = os.path.exists(train_stats_path)
    with open(train_stats_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'date', 'time', 'vocab_size', 'vector_size',
                'total_raw_words', 'effective_words', 'train_time_sec',
                'words_per_sec', 'alpha', 'model_path'
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            vocab_size,
            vector_size,
            raw_word_count,
            effective_word_count,
            train_time_sec,
            words_per_sec,
            alpha,
            model_save_path
    ])


default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='train_word2vec_from_sentiment_csv',
    default_args=default_args,
    schedule_interval=None,  # 필요 시 주기 설정
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['word2vec', 'nlp'],
) as dag:

    train_task = PythonOperator(
        task_id='train_word2vec_model',
        python_callable=train_word2vec
    )
