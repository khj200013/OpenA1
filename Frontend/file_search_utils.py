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
    # time.sleep(wait_sec)
    return vector_store.id

# 법 조항 찾아서 텍스트 분리
def extract_pipc_articles(text):
    chunks = re.split(r"[,<>\n]", text)
    current_law = None
    result = []

    for chunk in chunks:
        chunk = chunk.strip()

        match_combined = re.match(r"(?:개인\s*정보\s*보호\s*법|개인정보보호법)\s*제\s*(\d+)\s*조(?:\s*및\s*제\s*(\d+)\s*조)?", chunk)
        if match_combined:
            current_law = "개인정보보호법"
            result.append(f"{current_law}제{match_combined.group(1)}조")
            if match_combined.group(2):
                result.append(f"{current_law}제{match_combined.group(2)}조")
            continue
        
        # 법조항 여러개인 경우
        match_partial = re.match(r"제\s*(\d+)\s*조", chunk)
        if match_partial and current_law == "개인정보보호법":
            article = f"{current_law}제{match_partial.group(1)}조"
            result.append(article)
            continue

        match_number_only = re.match(r"(\d+)\s*조", chunk)
        if match_number_only and current_law == "개인정보보호법":
            article = f"{current_law}제{match_number_only.group(1)}조"
            result.append(article)
            

    return list(set(result))

# 전체 text 공백, 줄바꿈 제거
def clean_file_search_output(text):
    text = re.sub(r'<.*?>', '', text)

    text = text.strip()

    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    buffer_line = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+[\.\)]', line) or re.match(r'^[가-하][\.\)]', line):
            if buffer_line:
                cleaned_lines.append(buffer_line.strip())
                buffer_line = ''
            cleaned_lines.append(line)
        else:
            if buffer_line:
                buffer_line += ' ' + line
            else:
                buffer_line = line

    if buffer_line:
        cleaned_lines.append(buffer_line.strip())

    final_lines = []
    skip_next = False
    for i in range(len(cleaned_lines)):
        if skip_next:
            skip_next = False
            continue

        current_line = cleaned_lines[i].strip()
        if re.match(r'^\d+\.\s*$', current_line) and i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1].strip()
            final_lines.append(f"{current_line} {next_line}")
            skip_next = True
        else:
            final_lines.append(current_line)

    return '\n'.join(final_lines)

def file_search_query(client, user_input, vector_store_id):
    # user_input 내 개인정보보호버
    extract_context = extract_pipc_articles(user_input)

    if not extract_context :
        return "개인정보보호법 관련 법 조항이 없습니다."

    # 프롬프트 구성
    query = (
        "아래 질문에 있는 법 조항만 그대로 찾아줘.\n"
        "설명, 해석, 요약 없이 문장만 반환하고, 불필요한 서술을 덧붙이지 마.\n"
        f"질문: {extract_context}"
    )

    # 검색 요청
    response = client.responses.create(
        model="gpt-4o",
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
