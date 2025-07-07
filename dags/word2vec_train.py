from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv
import time
import os
import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pendulum

local_tz = pendulum.timezone("Asia/Seoul")


def train_word2vec():
    start_time = time.time()

    # 날짜별 파일 경로
    today_str = datetime.now().strftime("%Y%m%d")
    csv_path = f"/opt/airflow/data/blogjson_{today_str}/sentence_level_sentiment_with_scores.csv"
    model_save_path = "/opt/airflow/models/word2vec_custom.model"
    train_stats_path = "/opt/airflow/data/word2vec_train_stats.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    # 텍스트 전처리
    df = pd.read_csv(csv_path)
    texts = df['text'].dropna().drop_duplicates()

    okt = Okt()

    def tokenize_text(text):
        allowed_pos = ['Noun', 'Verb', 'Adjective']
        return [word for word, pos in okt.pos(text, stem=True) if pos in allowed_pos]

    tokenized_sentences = [tokenize_text(line.strip()) for line in texts if line.strip()]
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    total_token_count = len(all_tokens)  # 전체 단어 수 (중복 포함)

    # 모델 학습
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

    # 학습 통계
    train_time_sec = round(time.time() - start_time, 2)
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_time = datetime.now().strftime("%H:%M:%S")
    vocab_size = len(model.wv)
    vector_size = model.vector_size
    raw_word_count = model.corpus_total_words
    effective_word_count = model.corpus_count
    words_per_sec = int(effective_word_count / train_time_sec)
    alpha = model.alpha

    #  코사인 유사도 평균 계산 모델의 군집도 평가
    def average_cosine_similarity(model, words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) < 2:
            return 0.0
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = dot(vectors[i], vectors[j]) / (norm(vectors[i]) * norm(vectors[j]))
                similarities.append(sim)
        return round(np.mean(similarities), 4) if similarities else 0.0

    # 군집 평가를 위한 대표 단어 리스트 (예시: 음식 관련)
    common_words = ['음식', '강아지', '기쁨', '바다', '고통']
    avg_similarity = average_cosine_similarity(model, common_words)

    # ▒ CSV 로그 저장 ▒
    file_exists = os.path.exists(train_stats_path)
    with open(train_stats_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'date', 'time', 'vocab_size', 'vector_size',
                'total_raw_words', 'effective_words', 'total_token_count',
                'avg_similarity', 'train_time_sec', 'words_per_sec',
                'alpha', 'model_path'
            ])
        writer.writerow([
            train_date, train_time, vocab_size, vector_size,
            raw_word_count, effective_word_count, total_token_count,
            avg_similarity, train_time_sec, words_per_sec,
            alpha, model_save_path
        ])


# DAG 설정
default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='train_word2vec_with_similarity_logging',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['word2vec', 'nlp', 'embedding'],
) as dag:

    train_task = PythonOperator(
        task_id='train_word2vec_model',
        python_callable=train_word2vec
    )