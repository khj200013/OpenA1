# 표준 라이브러리
import os
import sys
import time

# 외부 라이브러리
import streamlit as st

# sys.path 추가(OpenA1\Main 폴더)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Main')))

# 내부 모듈
## 사이드바
from sidebar import render_history_sidebar, LABELS

## FileSearch 모듈
from file_search_utils import file_search_query, create_file, create_vector_store

## 유형 분류 모델 관련 모듈
from model_Frontend_v3 import classify_legal_issue, load_model

## UI렌더 관련 모듈
from ui_components import (
    render_user_input,
    render_user_message,
    render_assistant_message,
    render_assistant_message_stream,
    render_title_image,
    render_law_info,
)

## GPT(AI) 연동 관련 모듈
## law_info_ExceptDB -> GPT가 관련 법 조항을 찾고 내용 요약, 위반 가능성 설명
## action_guide -> 위반된 법 조항과 관련해 대응 절차 안내
from action_guide import action_guide_agent
from law_info_ExceptDB import get_law_info
from openai_utils import get_openai_client

# JSON 필터링 유틸 임포트 추가
from json_filtering import load_cases, filter_cases

# Client 설정
client = get_openai_client()

################################################################################
################################ Page Config ###################################
################################################################################


# Page Config 구성
st.set_page_config(
    page_title="Streamly - An Intelligent Streamlit Assistant",
    page_icon="imgs/avatar_streamly.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://github.com/khj200013/OpenA1/issues",
        "Report a bug": "https://github.com/khj200013/OpenA1/issues",
        "About": """
            ## AI를 통해 개인정보 보호법에 위배되는 사항 확인
            ### Powered using GPT-4

            **GitHub**: https://github.com/khj200013/OpenA1
            **문의 또는 버그 제보**: [Issue 등록하기](https://github.com/khj200013/OpenA1/issues)

            이 AI 어시스턴트는 사용자의 상황이나 질문을 바탕으로  
            **개인정보보호법 또는 관련 규정(GDPR 등)**에 따라  
            위반 가능성을 분석하고, 관련 법령과 대응 방안을 안내합니다.

            Image created by OpenAI
        """
    }
)


################################################################################
################################ GPT UI 구성 ###################################
################################################################################


# 로고 이미지 출력
render_title_image()

# 히스토리용 딕셔너리
if "history" not in st.session_state:
    st.session_state.history = { label: [] for label in LABELS }

# JSON 케이스 데이터 로드 (한 번만)
cases = load_cases()

# 질문→답변 생성 로직
if "user_input" in st.session_state:
    prompt = st.session_state.user_input

# 모델 로딩
if "model_loaded" not in st.session_state:
    with st.spinner("모델 및 벡터 스토어를 초기화하는 중입니다...잠시만 기다려주세요.."):
        tokenizer, model = load_model()
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.model_loaded = True

        # File Search용 설정
        file_path = "./개인정보보호법.pdf"
        file_id = create_file(client, file_path)
        vector_store_id = create_vector_store(client, file_id)

        # 설정 내용 session_state에 저장
        st.session_state.file_id = file_id
        st.session_state.vector_store = vector_store_id

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-4.1'

if 'messages' not in st.session_state:
    st.session_state.messages = []


# 기존 메시지 출력
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue  
    if msg["role"] == 'user':
        render_user_message(msg["content"])
    elif  msg["role"] == 'assistant': 
        if msg["law_info"]:
            render_law_info(msg["law_info"], msg["file_search_result"])
        render_assistant_message(msg["predicted_label"], msg["content"])


# 사용자 입력 저장 후 rerun 처리 (stream 중 입력 방지)
if "user_input" not in st.session_state:
    if prompt := render_user_input():
        st.session_state.user_input = prompt
        st.rerun()

# GPT 응답 처리
if "user_input" in st.session_state:
    prompt = st.session_state.user_input
    # 사용자 메시지 session_state에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 유저 메시지 렌더
    render_user_message(prompt)

    predict_start_time = time.time()

    # --------------------------
    # 1차 JSON 필터링 시도
    matched_case = filter_cases(prompt, cases)

    if matched_case is not None:
        # JSON 1차 필터링 결과 출력 및 히스토리 저장
        predicted_label = matched_case.get("label", "알 수 없음")
        law_article = matched_case.get("law_article", "")

    else:
        # JSON 매칭 안되면 기존 분류 + GPT 처리
        predicted_label = classify_legal_issue(prompt, st.session_state.tokenizer, st.session_state.model)
        law_article = None

    predict_end_time = time.time()

    print(f'라벨 분류 걸린 시간 :::: {predict_end_time-predict_start_time:.2f}초')

    # GPT PROMPT
    with st.spinner("🔎 법률 정보 분석 중입니다..."):
        law_info_start_time = time.time()


        law_info = get_law_info(predicted_label, prompt)

            
        law_info__end_time = time.time()

        print(f'법률정보 분석 걸린 시간 :::: {law_info__end_time-law_info_start_time:.2f}초')

        file_search_start_time = time.time()

        # File Search
        file_search_result = file_search_query(client, law_info['law'], st.session_state.vector_store)

        
        file_search__end_time = time.time()

        print(f'파일서치 걸린 시간 :::: {file_search__end_time-file_search_start_time:.2f}초')

    # Law info + File search 값 출력
    render_law_info(law_info, file_search_result)

    response_stream = action_guide_agent(prompt, predicted_label)

    # AI 메시지 렌더 (Stream 형태)
    response_text = render_assistant_message_stream(predicted_label, response_stream)
    
    # AI 메시지 session_state에 추가
    st.session_state.messages.append({"role": "assistant", 
                                      "predicted_label": predicted_label, 
                                      "law_info" : law_info,
                                      "file_search_result" : file_search_result,
                                      "content": response_text})
    # 질문 + 답변 저장
    st.session_state.history[predicted_label].append({
        "question": prompt,
        "answer": response_text
    })
    
    # user_input 삭제 후 rerun
    del st.session_state.user_input
    st.rerun()

################################################################################
################################[ 사 이 드 바 ]#################################
################################################################################

render_history_sidebar()

