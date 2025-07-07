import numpy as np
import re
import random
import time
from konlpy.tag import Okt
from gensim.models import Word2Vec

okt = Okt()

def filter_nouns(words):
    valid_nouns = set()
    for word in words:
        if re.fullmatch(r'[가-힣]{2,}', word):
            pos = okt.pos(word, stem=True, norm=True)
            if len(pos) == 1 and pos[0][1] == 'Noun':
                valid_nouns.add(word)
    return list(valid_nouns)

def quiz_by_similarity(word, model, max_vocab=50000):
    try:
        query_vec = model.wv[word]
    except KeyError:
        print(f"'{word}'는 모델에 존재하지 않습니다.")
        if input("다른 단어로 시도하시겠습니까? (Y/N): ").strip().lower() == 'y':
            new_word = input("새 단어 입력: ").strip()
            quiz_by_similarity(new_word, model)
        return

    print("단어 후보를 선정 중...")

    vocab_words = list(model.wv.key_to_index)[:max_vocab]
    candidate_words = [w for w in vocab_words if w != word]
    nouns = filter_nouns(candidate_words)

    if len(nouns) < 3:
        print("후보 명사가 부족합니다.")
        return

    vecs = model.wv[nouns]
    sims = np.dot(vecs, query_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10)

    bins = {'1.0~0.6': [], '0.6~0.4': [], '0.4~0.1': []}
    for w, s in zip(nouns, sims):
        if 0.6 <= s <= 1.0:
            bins['1.0~0.6'].append((w, s))
        elif 0.4 <= s < 0.6:
            bins['0.6~0.4'].append((w, s))
        elif 0.1 <= s < 0.4:
            bins['0.4~0.1'].append((w, s))

    selected = []
    for label in ['1.0~0.6', '0.6~0.4', '0.4~0.1']:
        selected.extend(sorted(bins[label], key=lambda x: -x[1])[:1])

    if len(selected) < 3:
        print("군집별로 충분한 단어를 찾지 못해 퀴즈 생성을 중단합니다.")
        return

    selected.append((word, 1.0))
    random.shuffle(selected)

    print(f"\n다음 중 '{word}'와 가장 유사한 단어는 무엇일까요?")
    for i, (w, _) in enumerate(selected, 1):
        print(f"{i}. {w}")

    # 시간 측정 시작
    start = time.time()

    try:
        choice = int(input("\n정답 번호를 선택하세요 (1~4): "))
        if not 1 <= choice <= len(selected):
            raise ValueError
    except ValueError:
        print("유효한 숫자를 입력해주세요.")
        return

    end = time.time()
    elapsed = round(end - start, 2)

    chosen_word, sim = selected[choice - 1]
    score = 1.0 if chosen_word == word else round(sim, 2)

    #  시간 점수 계산
    max_time = 90.0  # 기준 시간
    time_score = max(0.0, 1 - min(elapsed / max_time, 1))

    # 합산 점수 계산 (가중 평균)
    weight_score = 0.7
    weight_time = 0.3
    total_score = round(score * weight_score + time_score * weight_time, 4)

    print(f"\n선택한 단어: '{chosen_word}'")
    print(f"획득 점수: {score}")
    print(f"풀이 시간: {elapsed}초")
    print(f"시간 점수: {round(time_score, 4)}")
    print(f"합산 점수: {total_score}")

# 실행부
model = Word2Vec.load("models/word2vec_custom.model")
quiz_by_similarity(input("퀴즈 기준 단어를 입력하세요: ").strip(), model)