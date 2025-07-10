FROM apache/airflow:2.7.1-python3.9

USER root

RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    build-essential \
    curl \
 && apt-get clean

USER airflow

RUN pip install --no-cache-dir --upgrade pip python-dotenv
RUN pip install --no-cache-dir transformers[torch] scikit-learn optuna
RUN pip install --no-cache-dir numpy==1.24.4 scipy==1.10.1 gensim==4.3.2
RUN pip install --no-cache-dir fasttext konlpy torch fugashi[unidic-lite] accelerate
