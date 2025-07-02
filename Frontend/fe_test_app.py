# 표준 라이브러리
import os
import sys

# 외부 라이브러리
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# sys.path 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Main')))

# 내부 모듈
from model_Frontend_v2 import classify_legal_issue
from ui_components import (
    render_user_input,
    render_user_message,
    render_assistant_message,
    render_assistant_message_stream
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
        """
    }
)


################################################################################
################################ GPT UI 구성 ###################################
################################################################################


st.title("개인정보 법률 유형 분류 챗봇")

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-4.1'

if 'messages' not in st.session_state:
    st.session_state.messages = []


# 기존 메시지 출력
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue  
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 사용자 입력/출력 + GPT 응답 출력
if prompt := render_user_input():
    # 사용자 메시지 session_state에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 유저 메시지 렌더
    render_user_message(prompt)

    # 라벨 분류
    predicted_label = classify_legal_issue(prompt)

    # GPT PROMPT
    response_stream = action_guide_agent(prompt, predicted_label)

    # AI 메시지 렌더 (Stream 형태)
    response_text = render_assistant_message_stream(predicted_label, response_stream)
    
    # AI 메시지 session_state에 추가
    st.session_state.messages.append({"role": "assistant", "content": response_text})

################################################################################
################################[ 사 이 드 바 ]#################################
################################################################################

st.sidebar.empty() # 사이드바 초기화
with st.sidebar:
    # 상단 이미지 
    st.image('https://www.kamco.or.kr/portal/img/sub/01/emot1_new_001.png')

    # FAQ SelectBox
    violation_type = st.selectbox(
        '📌 분류 선택',
        (
            '1. 초상권 침해',
            '2. 동의 없는 개인정보 수집',
            '3. 목적 외 이용',
            '4. 제3자 무단 제공',
            '5. CCTV 과잉촬영',
            '6. 정보 유출',
            '7. 파기 미이행',
            '8. 광고성 정보 수신',
            '9. 계정/비밀번호 관련 문제',
            '10. 위치정보 수집/유출',
            '11. 개인정보 열람·정정 요구 거부',
        )
    )

    st.markdown(f"선택한 유형: **{violation_type}**")

