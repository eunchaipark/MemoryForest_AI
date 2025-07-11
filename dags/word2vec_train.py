from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv
import time
import os
import shutil
import pandas as pd
import optuna
from gensim.models import Word2Vec
from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pendulum

local_tz = pendulum.timezone("Asia/Seoul")


def train_word2vec(folder_date: str = None):
    start_time = time.time()

    if folder_date is None:
        folder_date = datetime.now().strftime("%Y%m%d")

    csv_path = f"/opt/airflow/data/sentiment_results/{folder_date}/sentence_level_sentiment_with_scores.csv"
    model_save_path = "/opt/airflow/models/word2vec_custom.model"
    backup_model_path = "/opt/airflow/models/word2vec_custom_backup.model"
    train_stats_path = "/opt/airflow/data/word2vec_train_stats.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df['text'].dropna().drop_duplicates()

    okt = Okt()

    def tokenize_text(text):
        allowed_pos = ['Noun', 'Verb', 'Adjective']
        return [word for word, pos in okt.pos(text, stem=True) if pos in allowed_pos]

    tokenized_sentences = [tokenize_text(line.strip()) for line in texts if line.strip()]
    all_tokens = [token for sentence in tokenized_sentences for token in sentence]
    total_token_count = len(all_tokens)

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

    common_words = ['음식', '강아지', '기쁨', '바다', '고통','손자','손녀','꽃', '자동차','사과','축구','초등학교','양복','회사원']
    baseline_similarity = 0.0
    restored = False

    if os.path.exists(model_save_path):
        baseline_model = Word2Vec.load(model_save_path)
        baseline_similarity = average_cosine_similarity(baseline_model, common_words)
        print(f"[기준 모델 평균 유사도]: {baseline_similarity}")
    else:
        print("[기존 모델 없음 - 초기 학습]")

    def objective(trial):
        vector_size = trial.suggest_categorical("vector_size", [50, 100, 150])
        window = trial.suggest_int("window", 3, 8)
        min_count = trial.suggest_int("min_count", 3, 10)
        epochs = trial.suggest_int("epochs", 5, 15)

        model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=epochs
        )
        score = average_cosine_similarity(model, common_words)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print(f"[최적 파라미터]: {best_params}")

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=best_params["vector_size"],
        window=best_params["window"],
        min_count=best_params["min_count"],
        workers=4,
        sg=1,
        epochs=best_params["epochs"]
    )
    model.save(model_save_path)
    print(f"Word2Vec 모델 저장 완료: {model_save_path}")

    new_similarity = average_cosine_similarity(model, common_words)
    print(f"[새 모델 평균 유사도]: {new_similarity}")

    if new_similarity < baseline_similarity:
        print("[성능 저하] 백업 모델로 복원")
        if os.path.exists(backup_model_path):
            if os.path.exists(model_save_path):
                if os.path.isdir(model_save_path):
                    shutil.rmtree(model_save_path)
                else:
                    os.remove(model_save_path)
            shutil.copy2(backup_model_path, model_save_path)
            restored = True
    else:
        print("[성능 향상 또는 유지] 백업 모델 갱신")
        if os.path.exists(backup_model_path):
            if os.path.isdir(backup_model_path):
                shutil.rmtree(backup_model_path)
            else:
                os.remove(backup_model_path)
        shutil.copy2(model_save_path, backup_model_path)

    train_time_sec = round(time.time() - start_time, 2)
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_time = datetime.now().strftime("%H:%M:%S")
    vocab_size = len(model.wv)
    vector_size = model.vector_size
    raw_word_count = model.corpus_total_words
    effective_word_count = model.corpus_count
    words_per_sec = int(effective_word_count / train_time_sec)
    alpha = model.alpha

    file_exists = os.path.exists(train_stats_path)
    with open(train_stats_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'date', 'time', 'vocab_size', 'vector_size',
                'total_raw_words', 'effective_words', 'total_token_count',
                'avg_similarity', 'train_time_sec', 'words_per_sec',
                'alpha', 'model_path', 'restored'
            ])
        writer.writerow([
            train_date, train_time, vocab_size, vector_size,
            raw_word_count, effective_word_count, total_token_count,
            new_similarity if not restored else baseline_similarity,
            train_time_sec, words_per_sec,
            alpha, model_save_path,
            'yes' if restored else 'no'
        ])


# DAG 설정
default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='train_word2vec_with_optuna_tuning',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['word2vec', 'nlp', 'optuna'],
) as dag:

    train_task = PythonOperator(
        task_id='train_word2vec_model',
        python_callable=train_word2vec,
        op_kwargs={'folder_date': datetime.now().strftime('%Y%m%d')}
    )
