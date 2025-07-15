from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pendulum import timezone
import urllib.request
import urllib.parse
import json
import os
import pendulum

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


# 카테고리 (주제별)
# 가족, 여행, 감정, 애완동물, 음식, 일상, 스포츠, 일상용품, 가전제품, 식물, 동물,
# 종교, 의료, 의상, 추억, 취미생활, 계절/날씨, 학교/교육, 직업/노동, 지역/고향,
# 교통/탈것, 건축/장소, 문화/예술, 기념일/행사, 어린 시절/육아



def crawl_blog_texts():
    node = 'blog'

    category_groups = {
       "감정" :
['기쁨', '웃음', '눈물', '외로움', '뿌듯함', '안도감', '걱정', '슬픔', '사랑', '설렘', '감동', '그리움', '허탈감', '추억', '희망', '정', '위로', '고마움', '반가움', '미소', '두려움', '따뜻함', '편안함', '불안', '긴장', '놀람', '공허함', '그윽함', '포근함', '친숙함']

    }

    today_str = datetime.now().strftime("%Y%m%d")

    for group, categories in category_groups.items():
        group_base_dir = os.path.join("/opt/airflow/data", group, today_str)
        os.makedirs(group_base_dir, exist_ok=True)

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

            save_path = os.path.join(group_base_dir, f"{category}_blog.json")
            with open(save_path, 'w', encoding='utf8') as outfile:
                json.dump(jsonResult, outfile, indent=4, ensure_ascii=False)
            print(f"[{group}] {category} 저장 완료: {save_path}")

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
