import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import random
import numpy as np
import os

################################################################################
################################[ API KEY 로드 ]################################
################################################################################

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

################################################################################
################################[ 함 수 정 의 ]#################################
################################################################################

# 피드백 함수
def save_feedback(index):
    st.session_state.messages[index]["feedback"] = st.session_state[f"feedback_{index}"]

################################################################################
################################[ 초 기 설 정 ]#################################
################################################################################

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = 'gpt-4.1'

if 'messages' not in st.session_state:
    st.session_state.messages = []

################################################################################
################################[ Page Config ]#################################
################################################################################

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
##################################[ 채 팅 ]#####################################
################################################################################

st.title('👨‍⚖️ 개인정보 법률 유형 분류 챗봇 👩‍⚖️')
st.write('개인정보 보호법에 위반되는지 판별하고 대응방안을 확인하세요!')


# 기존 메시지 출력
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        continue  
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg["role"] == "assistant":
            feedback = msg.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )


# 사용자 입력
if prompt := st.chat_input('개인정보 보호에 관한 법률 문제를 입력하세요'):

    # 대화 내용 messages 추가
    st.session_state.messages.append({
        "role":"user",
        "content": prompt
    })
    
    # 사용자 입력 화면 출력
    with st.chat_message('user'):
        st.markdown(prompt)

    # 답변 화면 출력
    with st.chat_message('assistant'):
        stream = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[
                {"role" : m['role'], "content":m['content']}
                for m in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.messages)}",
            on_change=save_feedback,
            args=[len(st.session_state.messages)]
        )

    # 메시지에 추가
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )


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
            '9. 계정 관련 문제',
            '10. 정보열람/철회권 거부'
        )
    )

    st.markdown(f"선택한 유형: **{violation_type}**")

