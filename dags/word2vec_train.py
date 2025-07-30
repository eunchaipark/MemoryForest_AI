from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv, time, os, shutil, random
import pandas as pd
import optuna
from gensim.models import Word2Vec
from konlpy.tag import Okt
import numpy as np
import pendulum
import re
from numpy import dot
from numpy.linalg import norm

local_tz = pendulum.timezone("Asia/Seoul")
SIMILARITY_DROP_THRESHOLD = 0.01
MAX_RETRIES = 3
BATCH_SIZE = 100           # 최대 문장 수
MAX_BATCH_TOKENS = 1200    # 배치 내 최대 토큰 총량 (조절 가능)

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
    texts = df['text'].dropna().drop_duplicates().tolist()

    stopwords = {
        "이", "그", "저", "것", "게", "내", "너", "나", "우리", "뭐", "왜", "어디", "어떻게",
        "그리고", "그러나", "에서", "으로", "까지", "부터", "보다", "하고", "의", "에", "와", "과",
        "는", "은", "이", "가",
        "이다", "입니다", "있다", "합니다", "않다", "않습니다",
        "도", "만", "뿐", "처럼", "으로서", "이나", "든가", "라도",
        "다", "다니", "다고", "지만", "는데", "으면", "니까", "면서",
        "아", "야", "어", "음", "흠", "휴", "헉", "와",
        "어쩌구", "그런데", "아무튼", "아마"
    }

    okt = Okt()

    def clean_text(text):
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_text(text):
        tokens = okt.morphs(text, stem=True)
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    # 전체 문장 토큰화 + 전처리 (중복, 길이 체크 포함)
    processed_sentences = []
    seen_sentences = set()
    for line in texts:
        line = clean_text(line)
        if line and line not in seen_sentences and len(line) >= 10:
            seen_sentences.add(line)
            tokens = tokenize_text(line)
            if len(tokens) >= 3:
                processed_sentences.append(tokens)

    total_token_count = sum(len(s) for s in processed_sentences)

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

    # Optuna 최적 파라미터 탐색
    def objective(trial):
        vector_size = trial.suggest_categorical("vector_size", [100, 200, 300])
        window = trial.suggest_int("window", 2, 10)
        min_count = trial.suggest_int("min_count", 2, 20)
        epochs = trial.suggest_int("epochs", 10, 40)
        model = Word2Vec(
            sentences=processed_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=epochs,
            seed=42
        )
        return average_cosine_similarity(model, common_words)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f"[최적 파라미터]: {best_params}")

    # 초기 모델 준비
    if os.path.exists(model_save_path):
        model = Word2Vec.load(model_save_path)
        baseline_similarity = average_cosine_similarity(model, common_words)
        print(f"[기존 모델 평균 유사도]: {baseline_similarity}")
    else:
        # 새 모델 초기화
        model = Word2Vec(
            vector_size=best_params["vector_size"],
            window=best_params["window"],
            min_count=best_params["min_count"],
            workers=4,
            sg=1,
            epochs=best_params["epochs"],
            seed=42
        )
        baseline_similarity = 0.0
        print("[기존 모델 없음 - 새 모델 생성]")

    best_similarity = baseline_similarity
    best_model = model
    restored = False

    # 배치 생성 함수 (문장 수 + 토큰 수 제한 혼합)
    def batch_sentences_with_token_limit(sentences, max_sentences, max_tokens):
        batch = []
        token_count = 0

        for sent in sentences:
            sent_len = len(sent)
            # 문장 하나가 너무 길면 단독 배치 처리
            if sent_len > max_tokens:
                if batch:
                    yield batch
                    batch = []
                    token_count = 0
                yield [sent]
                continue

            if len(batch) >= max_sentences or token_count + sent_len > max_tokens:
                yield batch
                batch = []
                token_count = 0

            batch.append(sent)
            token_count += sent_len

        if batch:
            yield batch

    initial_alpha = 0.025
    min_alpha = 0.0001

    for batch_idx, batch_sentences in enumerate(batch_sentences_with_token_limit(processed_sentences, BATCH_SIZE, MAX_BATCH_TOKENS), start=1):
        print(f"\n[배치 {batch_idx} - 문장 수: {len(batch_sentences)} 토큰 수: {sum(len(s) for s in batch_sentences)}] 학습 시작")

        random.shuffle(batch_sentences)

        batch_best_similarity = -1.0
        batch_model = None
        alpha = initial_alpha

        for retry in range(1, MAX_RETRIES + 1):
            print(f"  [시도 {retry}/{MAX_RETRIES}]")

            if retry == 1 and batch_idx == 1 and baseline_similarity == 0.0:
                model.build_vocab(batch_sentences)
            else:
                model.build_vocab(batch_sentences, update=True)

            current_min_alpha = max(min_alpha, alpha * 0.1)

            model.train(
                batch_sentences,
                total_examples=len(batch_sentences),
                epochs=best_params["epochs"],
                start_alpha=alpha,
                end_alpha=current_min_alpha
            )

            sim = average_cosine_similarity(model, common_words)
            print(f"    [학습 후 유사도]: {sim:.4f}")

            if sim > batch_best_similarity + 1e-4:
                batch_best_similarity = sim
                batch_model = model
                print("    [성능 향상됨 → 배치 모델 갱신]")
                alpha *= 0.8
            else:
                print("    [성능 향상 없음 → 시도 종료]")
                break

        if batch_best_similarity > best_similarity + 1e-4:
            best_similarity = batch_best_similarity
            best_model = batch_model
            print(f"[배치 {batch_idx}] 모델 성능 개선됨 → 모델 유지")
        else:
            print(f"[배치 {batch_idx}] 모델 성능 개선 없음 → 배치 데이터 버리고 다음 배치로")
            # vocab revert 어려움 -> 학습 효과 없으면 무시

    similarity_gap = baseline_similarity - best_similarity
    print(f"\n[최종 유사도 변화량]: {similarity_gap:.4f}")

    if baseline_similarity > 0 and similarity_gap > SIMILARITY_DROP_THRESHOLD:
        print("[성능 저하 감지 - 백업 모델로 복원]")
        if os.path.exists(backup_model_path):
            shutil.copy2(backup_model_path, model_save_path)
            restored = True
    else:
        print("[성능 유지 또는 향상 - 최종 모델 저장 및 백업 갱신]")
        best_model.save(model_save_path)
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
            train_date, train_time, len(best_model.wv.key_to_index), best_model.vector_size,
            best_model.corpus_total_words, best_model.corpus_count, total_token_count,
            best_similarity if not restored else baseline_similarity,
            train_time_sec, int(best_model.corpus_count / train_time_sec),
            best_model.alpha, model_save_path,
            'yes' if restored else 'no'
        ])

# DAG 정의
default_args = {
    'start_date': datetime(2025, 6, 17),
    'catchup': False,
}

with DAG(
    dag_id='train_word2vec_with_optuna_and_retries',
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
