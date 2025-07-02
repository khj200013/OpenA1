import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 모델과 토크나이저 불러오기
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("./saved_klue_bert2")
    model = BertForSequenceClassification.from_pretrained("./saved_klue_bert2")
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
    7: "계정/비밀번호 관련 문제",
    8: "개인정보 열람·정정 요구 거부"
}

# 2. 분류 함수
def classify_legal_issue(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

# 3. Streamlit UI 구성
st.title("개인정보 법률 유형 분류 챗봇")

user_input = st.text_area("법률 문제 내용을 입력하세요", height=150)

if st.button("분류하기"):
    if user_input.strip() == "":
        st.warning("내용을 입력해 주세요.")
    else:
        result = classify_legal_issue(user_input, tokenizer, model)
        st.success(f"예측된 유형: {result}")

        guide_messages = {
            "초상권 침해": "→ 초상권 침해 관련 법률 조치를 확인하세요.",
            "동의 없는 개인정보 수집": "→ 개인정보 수집 시 동의 절차가 있었는지 확인하세요.",
            "목적 외 이용": "→ 수집 목적 외 사용은 위법 소지가 있습니다.",
            "제3자 무단 제공": "→ 제3자 제공 여부 및 동의 기록을 확인하세요.",
            "CCTV 과잉촬영": "→ CCTV 설치 목적 및 고지 여부를 확인하세요.",
            "정보 유출": "→ 유출 경로 확인 및 신고를 고려해 보세요.",
            "파기 미이행": "→ 보관 기간 종료 후 파기 조치가 되었는지 확인하세요.",
            "계정/비밀번호 관련 문제": "→ 비밀번호 유출 여부 확인 및 즉시 변경이 필요합니다.",
            "개인정보 열람·정정 요구 거부": "→ 열람·정정 요청이 정당한지 확인하고 대응 방안을 모색하세요."
        }

        st.info(guide_messages.get(result, "→ 추가 조치가 필요합니다."))


