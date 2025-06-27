from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pendulum import timezone
import urllib.request
import urllib.parse
import json
import os
import pendulum

# 6월 17일을 기점으로 
# ['일상', '여행', '가족', '음식', '건강', '산책', '취미', '기억력', '인지훈련', '웃음', '우울', '행복', '공허', '허탈', '애완', '사랑', '추억', '사진', '기억', '친구', '이웃', '감정', '위로', '희망', '설렘', '감동', '소중함', '그리움', '자녀', '부모님', '학교', '옛날', '젊음', '첫 경험', '즐거움', '모험', '운동', '만남', '가족모임', '손자', '손녀', '편안함', '자연', '새벽', '노을', '별', '감성']
# 에 대한 블로그 텍스트 가져옴
# 감각자극 관련 키워드  6월 27일 기준 변환
# ['바람', '향기', '빗소리', '따뜻함', '촉감', '햇살', '눈물', '온기', '새소리', '정원', '꽃향기']

# 일상 활동 루틴관련키워드
# ['빨래', '장보기', '요리', '반찬', '청소', '목욕', '텃밭', '일기', '라디오', '텔레비전']

#사회적 관계 및 정서 키워드
# ['친구들', '동네', '반장', '교회', '시장', '동창', '이모', '삼촌', '옆집', '강아지', '고양이']

# 시대 문화 회상 관련 키워드
# ['흑백사진', '교복', '국민학교', '종이학', '쌍쌍바', '만화책', '옛동요', '가요무대', '라디오 사연', '고무줄놀이', '굴렁쇠']

#자연, 계절, 풍경 관련 키둬
# ['봄', '여름', '가을', '겨울', '단풍', '눈', '장마', '들꽃', '바다', '숲길']



client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")
local_tz = pendulum.timezone("Asia/Seoul")


def getRequestUrl(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print(f"[{datetime.now()}] Request Success")
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error for URL : {url}\n{e}")
        return None

def getNaverSearch(node, srcText, start, display):
    base = "https://openapi.naver.com/v1/search"
    node = f"/{node}.json"
    parameters = f"?query={urllib.parse.quote(srcText)}&start={start}&display={display}"
    url = base + node + parameters
    response = getRequestUrl(url)
    return json.loads(response) if response else None

def getPostData(post, jsonResult, cnt):
    jsonResult.append({
        'cnt': cnt,
        'title': post['title'],
        'description': post['description'],
        'link': post['link'],
        'pDate': datetime.strptime(post['postdate'], '%Y%m%d').strftime('%Y-%m-%d')
    })

def crawl_blog_texts():
    node = 'blog'
    categories = ['일상', '여행', '가족', '음식', '건강', '산책', '취미', '기억력', '인지훈련', '웃음', '우울', '행복', '공허', '허탈', '애완', '사랑', '추억', '사진', '기억', '친구', '이웃', '감정', '위로', '희망', '설렘', '감동', '소중함', '그리움', '자녀', '부모님', '학교', '옛날', '젊음', '첫 경험', '즐거움', '모험', '운동', '만남', '가족모임', '손자', '손녀', '편안함', '자연', '새벽', '노을', '별', '감성']
    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = f"/opt/airflow/data/blogjson_{today_str}"  # <- 경로 변경
    os.makedirs(save_dir, exist_ok=True)

    for category in categories:
        cnt = 0
        jsonResult = []
        jsonResponse = getNaverSearch(node, category, 1, 100)
        if not jsonResponse:
            continue

        total = jsonResponse['total']
        while jsonResponse and jsonResponse['display'] != 0 and cnt < 1000:
            for post in jsonResponse['items']:
                cnt += 1
                getPostData(post, jsonResult, cnt)
            start = jsonResponse['start'] + jsonResponse['display']
            jsonResponse = getNaverSearch(node, category, start, 100)

        save_path = os.path.join(save_dir, f"{category}_blog.json")
        with open(save_path, 'w', encoding='utf8') as outfile:
            json.dump(jsonResult, outfile, indent=4, ensure_ascii=False)
        print(f"{category} 저장 완료: {save_path}")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 13),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_blog_crawling',
    default_args=default_args,
    schedule_interval=None,
    start_date=pendulum.datetime(2025, 6, 17, tz=local_tz),
    catchup=False,
    tags=['naver', 'blog', 'crawl']
) as dag:
    crawl_task = PythonOperator(
        task_id='crawl_blog_texts',
        python_callable=crawl_blog_texts,
    )
