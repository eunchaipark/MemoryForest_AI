from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv, time, os, shutil
import pandas as pd
import optuna
from gensim.models import Word2Vec
from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pendulum

local_tz = pendulum.timezone("Asia/Seoul")
SIMILARITY_DROP_THRESHOLD = 0.01  # 유사도 1% 이상 감소 시만 복원

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

    stopwords = [ "이", "그", "저", "것", "게", "내", "너", "나", "우리", "뭐", "왜", "어디", "어떻게",
                  "또한", "그리고", "그러나", "그래서", "좀", "같이", "자주", "매우", "너무", "결국", "사실",
                  "에서", "으로", "까지", "부터", "보다", "하고", "의", "에", "와", "과", "는", "은", "이", "가",
                  "한다", "했다", "하게", "하여", "하는", "되다", "된다", "되며", "이다", "있는", "없는", "하였다" ]

    okt = Okt()

    def tokenize_text(text):
        allowed_pos = ['Noun', 'Verb', 'Adjective']
        tokens = [word for word, pos in okt.pos(text, stem=True) if pos in allowed_pos]
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    seen_sentences = set()
    tokenized_sentences = []

    for line in texts:
        line = line.strip()
        if line and line not in seen_sentences:
            seen_sentences.add(line)
            tokens = tokenize_text(line)
            if len(tokens) >= 3:
                tokenized_sentences.append(tokens)

    total_token_count = sum(len(s) for s in tokenized_sentences)

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

    common_words = [
        "부모", "유럽", "기쁨", "고양이", "김치찌개", "출근", "축구", "칫솔",
        "냉장고", "선인장", "호랑이", "불교", "병원", "청바지", "졸업식", "독서",
        "봄비", "교과서", "간호사", "부산", "지하철", "박물관", "연극", "생일", "유치원"
    ]

    baseline_similarity = 0.0
    restored = False

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
        return average_cosine_similarity(model, common_words)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"[최적 파라미터]: {best_params}")

    # 모델 불러오기 + 점진적 학습
    if os.path.exists(model_save_path):
        model = Word2Vec.load(model_save_path)
        baseline_similarity = average_cosine_similarity(model, common_words)
        print(f"[기존 모델 평균 유사도]: {baseline_similarity}")
        model.build_vocab(tokenized_sentences, update=True)
        model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=best_params["epochs"])
    else:
        print("[기존 모델 없음 - 새로 학습 시작]")
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
    similarity_gap = baseline_similarity - new_similarity
    print(f"[새 모델 평균 유사도]: {new_similarity}")
    print(f"[유사도 변화량]: {similarity_gap:.4f}")

    if baseline_similarity > 0 and similarity_gap > SIMILARITY_DROP_THRESHOLD:
        print("[ 성능 저하] 백업 모델로 복원")
        if os.path.exists(backup_model_path):
            shutil.copy2(backup_model_path, model_save_path)
            restored = True
    else:
        print("[ 성능 유지 또는 허용 수준의 저하] 백업 모델 갱신")
        shutil.copy2(model_save_path, backup_model_path)

    train_time_sec = round(time.time() - start_time, 2)
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_time = datetime.now().strftime("%H:%M:%S")

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
            train_date, train_time, len(model.wv.key_to_index), model.vector_size,
            model.corpus_total_words, model.corpus_count, total_token_count,
            new_similarity if not restored else baseline_similarity,
            train_time_sec, int(model.corpus_count / train_time_sec),
            model.alpha, model_save_path,
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
