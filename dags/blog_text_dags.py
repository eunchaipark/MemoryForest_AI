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

def crawl_blog_texts():
    node = 'blog'
    categories =['이발기', '비닐우산', '유리병', '빨래비누', '빨래줄', '호롱불', '석유난로', '연탄난로', '휴대용라디오', '풍로', '숟가락통', '옷솔', '비닐우의', '꽃무늬이불', '나무장롱', '손때묻은문고리', '지우개', '필통', '잉크병', '만년필', '종이인형', '방울토마토', '텃밭삽', '마늘망', '고추꼭지', '모래놀이터', '철봉', '시소', '그네', '회전기구', '놀이터그림자', '옛날미끄럼틀', '도토리', '밤송이', '물수제비', '돌팔매질', '진흙탕', '빈깡통차기', '비석치기', '자치기', '딱지치기', '오징어놀이', '공기돌', '구슬자국', '콩주머니', '비눗방울', '맷돌', '지렁이', '개울돌', '연잎']


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
