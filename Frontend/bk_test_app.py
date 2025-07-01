import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import torch, os
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 모델과 토크나이저 불러오기
@st.cache_resource  # 캐싱해서 재실행 시 빠르게
def load_model():
    # tokenizer = BertTokenizer.from_pretrained("./saved_klue_bert1")
    # model = BertForSequenceClassification.from_pretrained("./saved_klue_bert1")
    tokenizer = BertTokenizer.from_pretrained("../../saved_klue_bert")
    model = BertForSequenceClassification.from_pretrained("../../saved_klue_bert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

label_map = {
    0: "초상권 침해",
    1: "동의 없는 개인정보 수집",
    2: "목적 외 이용",
    3: "제3자 무단 제공",
    4: "CCTV 과잉촬영",
    5: "정보 유출",
    6: "파기 미이행",
    7: "광고성 정보 수신",
    8: "계정/비밀번호 관련 문제"
}

# 2. 분류 함수
def classify_legal_issue(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

# 피드백 함수
def save_feedback(index):
    st.session_state.messages[index]["feedback"] = st.session_state[f"feedback_{index}"]


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



# GPT AI 구성
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)



# 3. Streamlit UI 구성
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



if prompt := st.chat_input("개인정보 보호에 관한 법률 문제를 입력하세요"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    predicted_label = classify_legal_issue(prompt)

    with st.chat_message("assistant"):
        st.markdown(f"🔍 **예측된 유형:** {predicted_label}")
        st.markdown("AI 상담 내용을 불러오는 중입니다...")

        gpt_prompt = f"""
        다음은 개인정보 침해와 관련된 사용자 질문입니다:
        사용자 질문: "{prompt}"
        예측된 법 위반 유형: {predicted_label}
        이에 따라 가능한 법적 설명과 대응 방법을 안내해 주세요.
        """

        stream = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": "system", "content": "당신은 개인정보보호 전문가입니다."},
                {"role": "user", "content": gpt_prompt}
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

    
    st.session_state.messages.append({"role": "assistant", "content": response})

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

