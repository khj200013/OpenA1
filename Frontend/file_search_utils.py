import requests
from openai import OpenAI
from io import BytesIO
import time
import re


def create_file(client, file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        result = client.files.create(file=file_tuple, purpose="assistants")
    else:
        with open(file_path, "rb") as file_content:
            result = client.files.create(file=file_content, purpose="assistants")
    return result.id

def create_vector_store(client, file_id, wait_sec=15):
    # 벡터스토어 만들기 & 파일 연결
    vector_store = client.vector_stores.create(name='knowledge_base')
    client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file_id)
    
    # 등록 완료까지 대기
    time.sleep(wait_sec)
    return vector_store.id


def extract_pipc_articles(text):
    chunks = re.split(r"[,<>\n]", text)
    current_law = None
    result = []

    for chunk in chunks:
        chunk = chunk.strip()

        match_full = re.match(r"(개인정보보호법)\s*제\s*(\d+)\s*조", chunk)
        if match_full:
            current_law = match_full.group(1)
            article = f"{current_law}제{match_full.group(2)}조"
            result.append(article)
            continue

        # "제26조"처럼 앞에 법령이 생략된 경우
        match_partial = re.match(r"제\s*(\d+)\s*조", chunk)
        if match_partial and current_law == "개인정보보호법":
            article = f"{current_law}제{match_partial.group(1)}조"
            result.append(article)

    return result

def clean_file_search_output(text):
    text = re.sub(r'<.*?>', '', text)

    text = text.strip()

    text = re.sub(r'\n\s*\n+', '\n\n', text)

    lines = text.split('\n')
    merged_lines = []
    skip_next = False

    for i in range(len(lines)):
        if skip_next:
            skip_next = False
            continue

        current_line = lines[i].strip()

        if re.match(r'^\d+\.\s*$', current_line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            merged_lines.append(f"{current_line} {next_line}")
            skip_next = True
        else:
            merged_lines.append(current_line)

    return '\n'.join(merged_lines)

def file_search_query(client, user_input, vector_store_id):
    extract_context = extract_pipc_articles(user_input)
    print(extract_context)

    # 프롬프트 구성
    query = (
        "아래 질문에 있는 법 조항만 파일 내에서 찾아서 설명 없이 문장만 반환해.\n"
        # "만약 관련 내용이 없으면 해당 조항에 영향을 끼치는 조항을 찾아서 출력해줘\n"
        "만약 관련 내용이 없으면 \'관련 법 조항이 개인정보보호법에 없습니다.\' 라고 말해줘"
        f"질문: {extract_context}"
    )

    # 검색 요청
    response = client.responses.create(
        model="gpt-4.1",
        input=query,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }],
        tool_choice="required"
    )

    # 응답 파싱
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content_item in item.content:
                if getattr(content_item, "type", None) == "output_text":
                    raw_text = content_item.text
                    # 파일 여백, 공백 처리
                    result = clean_file_search_output(raw_text)
                    return result

    return "결과 없음"
