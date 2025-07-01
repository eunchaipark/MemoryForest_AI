import numpy as np
import re
import random
from konlpy.tag import Okt
from gensim.models import Word2Vec

okt = Okt()

def is_korean_noun(word):
    if not re.fullmatch(r'[가-힣]{2,}', word):
        return False
    pos = okt.pos(word, stem=True)
    return len(pos) == 1 and pos[0][1] == 'Noun'

def cosine_similarity_matrix(query_vec, mat):
    query_norm = np.linalg.norm(query_vec)
    mat_norms = np.linalg.norm(mat, axis=1)
    dot_products = mat @ query_vec
    sim = dot_products / (mat_norms * query_norm + 1e-10)
    return sim

def quiz_by_similarity(word, model, max_vocab=50000):
    try:
        query_vec = model.wv[word]
    except KeyError:
        print(f"'{word}'는 단어 유사도 출력 모델에 없습니다.")
        yes_no = input("다른 단어를 입력해주세요 Y or N : ")
        if(yes_no == 'Y' or 'y : '):
            input_word = input("퀴즈 생성 기준 단어를 입력하세요: ")
            quiz_by_similarity(input_word.strip(), model)
        return

    print("단어 후보를 선정 중...")

    words = list(model.wv.key_to_index.keys())[:max_vocab]
    nouns = [w for w in words if w != word and is_korean_noun(w)]
    vecs = np.array([model.wv[w] for w in nouns])
    sims = cosine_similarity_matrix(query_vec, vecs)

    bins = {
        '1.0~0.6': [],
        '0.6~0.4': [],
        '0.4~0.1': [],
    }

    for w, s in zip(nouns, sims):
        if 0.6 <= s <= 1.0:
            bins['1.0~0.6'].append((w, s))
        elif 0.4 <= s < 0.6:
            bins['0.6~0.4'].append((w, s))
        elif 0.1 <= s < 0.4:
            bins['0.4~0.1'].append((w, s))

    desired = {'1.0~0.6': 1, '0.6~0.4': 1, '0.4~0.1': 1}
    selected = []

    for label in ['1.0~0.6', '0.6~0.4', '0.4~0.1']:
        available = sorted(bins[label], key=lambda x: x[1], reverse=True)
        selected.extend(available[:desired[label]])

    if len(selected) < 3:
        print("충분한 단어가 없어 퀴즈를 생성할 수 없습니다. ")
        return

    selected.append((word, 1.0))  # 입력 단어 자체도 포함

    random.shuffle(selected)
    print(f"\n다음 중 '{word}'와 가장 의미가 같은 단어를 고르세요:")
    for i, (w, _) in enumerate(selected, 1):
        print(f"{i}. {w}")

    try:
        choice = int(input("\n정답 번호를 선택하세요 (1~4): "))
        if not 1 <= choice <= len(selected):
            raise ValueError
    except ValueError:
        print("잘못된 입력입니다. 숫자 1~4를 입력하세요.")
        return

    chosen_word, sim = selected[choice - 1]
    score = 1.0 if chosen_word == word else round(sim, 2)

    print(f"\n당신의 선택: '{chosen_word}'")
    print(f"획득 점수: {score}")


model_path = "models/word2vec_custom.model"
model = Word2Vec.load(model_path)

input_word = input("퀴즈 생성 기준 단어를 입력하세요: ")
quiz_by_similarity(input_word.strip(), model)