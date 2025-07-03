# 표준 라이브러리
import os
import sys
import time

# 외부 라이브러리
import streamlit as st

# sys.path 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Main')))

# 내부 모듈
from model_Frontend_v3 import classify_legal_issue, load_model
from ui_components import (
    render_user_input,
    render_user_message,
    render_assistant_message,
    render_assistant_message_stream,
    render_title_image,
)
from action_guide import action_guide_agent


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

# 모델 로딩
if "model_loaded" not in st.session_state:
    tokenizer, model = load_model()
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model
    st.session_state.model_loaded = True


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

    # 라벨 분류
    predicted_label = classify_legal_issue(prompt, st.session_state.tokenizer, st.session_state.model)

    # 분류된 라벨에 질문 저장.
    st.session_state.history[predicted_label].append(prompt)
    
    # GPT PROMPT
    response_stream = action_guide_agent(prompt, predicted_label)

    # AI 메시지 렌더 (Stream 형태)
    response_text = render_assistant_message_stream(predicted_label, response_stream)
    
    # AI 메시지 session_state에 추가
    st.session_state.messages.append({"role": "assistant", "predicted_label": predicted_label, "content": response_text})

    # user_input 삭제 후 rerun
    del st.session_state.user_input
    st.rerun()

################################################################################
################################[ 사 이 드 바 ]#################################
################################################################################

from sidebar import render_history_sidebar


# 히스토리용 딕셔너리
if "history" not in st.session_state:
    st.session_state.history = {
        "초상권 침해": [],
        "동의 없는 개인정보 수집": [],
        "목적 외 이용": [],
        "제3자 무단 제공": [],
        "CCTV 과잉촬영": [],
        "정보 유출": [],
        "파기 미이행": [],
        "광고성 정보 수신": [],
        "계정/비밀번호 관련 문제": [],
        "위치정보 수집/유출": [],
        "개인정보 열람·정정 요구 거부": []
    }

render_history_sidebar()

