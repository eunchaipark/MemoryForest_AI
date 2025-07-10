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
        "스포츠" : ["축구공", "농구골대", "탁구채", "배드민턴", "하키", "야구글러브", "승마", "배구", "러닝화", "수영복", "수영모", "구기운동", "심판", "득점", "스케이트보드", "롤러블레이드", "암벽등반", "줄넘기", "운동장", "심장박동", "스트레칭", "근육통", "운동복", "운동화", "조깅", "사이클", "체력단련", "도핑", "피트니스", "헬스장", "헬스기구", "바벨", "역도", "유도", "태권도", "합기도", "격투기", "배영", "자전거도로", "계영", "릴레이", "경기장", "기록", "응원석", "스탠드", "공격수", "수비수", "골키퍼", "득점판", "작전판"],

"의상" : ["셔츠", "블라우스", "후드티", "점퍼", "재킷", "드레스", "슬랙스", "청바지", "미니스커트", "맥시스커트", "가디건", "니트", "벨트", "타이", "보타이", "셔링", "프릴", "워커", "부츠", "로퍼", "뮬", "스니커즈", "슬리퍼", "코디", "스타일링", "스카프", "목도리", "비니", "페도라", "베레모", "양산", "클러치", "숄더백", "백팩", "팔찌", "귀걸이", "목걸이", "반지", "헤어핀", "스킨케어", "립스틱", "마스카라", "파운데이션", "향수", "런웨이", "패션쇼", "룩북", "트렌드", "패션매거진"],

"직업/노동" : ["출근길", "퇴근시간", "지각", "야근", "재택근무", "화상회의", "부서이동", "인사팀", "마케팅팀", "영업팀", "기획서", "보고서", "업무일지", "결재서류", "책상정리", "복합기", "사무용품", "사원증", "명함", "회의실", "출입카드", "연차", "반차", "연봉", "성과급", "성과평가", "승진", "발령", "부장", "차장", "과장", "대리", "주임", "사원", "상사", "부하직원", "업무분장", "직무교육", "기업문화", "회식", "동료", "상사피드백", "보고체계", "업무협업", "출장", "근태관리", "사내공지", "주간업무", "업무보고"],

# "어린 시절/육아" : ["기저귀", "젖병", "분유", "아기띠", "유모차", "수유", "이유식", "턱받이", "아기침대", "모빌", "딸랑이", "손수건", "아기옷", "베넷저고리", "속싸개", "물티슈", "체온계", "콧물흡입기", "유아세제", "아기욕조", "이유식조리기", "모유", "분유포트", "비누방울", "동화책", "퍼즐", "블록", "놀이매트", "아기체육관", "베이비샴푸", "비누망", "젖꼭지", "아기이불", "수면조끼", "기저귀통", "아기용로션", "소아과", "육아일기", "알림장", "어린이집", "유치원", "학부모회", "알림장가방", "유아카시트", "수면교육", "산모수첩", "배변훈련", "분리불안", "장난감박스", "아기모자"],

# "교통/탈것" : ["자동차", "오토바이", "자전거", "킥보드", "전동킥보드", "버스", "시내버스", "광역버스", "마을버스", "고속버스", "택시", "지하철", "기차", "KTX", "SRT", "전철", "승강장", "버스정류장", "지하철역", "호출앱", "내비게이션", "운전면허증", "차고지", "차키", "계기판", "핸들", "브레이크", "가속페달", "사이드미러", "백미러", "후방카메라", "비상등", "와이퍼", "타이어", "엔진오일", "주차장", "정차", "교통카드", "대중교통", "환승", "자가용", "운전연수", "차량번호판", "운전석", "조수석", "운전자보험", "하이패스", "탑승"]

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
