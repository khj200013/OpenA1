import streamlit as st
from pathlib import Path

LABELS = [
    "초상권 침해",
    "동의 없는 개인정보 수집",
    "제3자 무단 제공",
    "CCTV 과잉촬영",
    "정보 유출",
    "파기 미이행",
    "계정/비밀번호 관련 문제",
    "개인정보 열람·정정 요구 거부",
]

# 각 카테고리에 대한 간략 설명을 여기에 작성
CATEGORY_DESC = {
    "초상권 침해": "타인의 얼굴·신체 이미지가 허가 없이 사용되는 경우를 말합니다.",
    "동의 없는 개인정보 수집": "사용자의 동의를 받지 않고 개인정보를 수집하는 행위입니다.",
    "제3자 무단 제공": "개인정보를 동의 없이 외부에 제공하는 행위입니다.",
    "CCTV 과잉촬영": "필요 이상의 범위나 시간 동안 영상정보를 촬영하는 경우입니다.",
    "정보 유출": "보유 중인 개인정보가 외부로 유출된 사고를 의미합니다.",
    "파기 미이행": "보유기간 종료 후 개인정보를 제대로 파기하지 않은 경우입니다.",
    "계정/비밀번호 관련 문제": "로그인 정보 관리 및 유출 문제를 포함합니다.",
    "개인정보 열람·정정 요구 거부": "정보주체의 열람·정정 요구를 거부하는 행위입니다.",
}


def render_history_sidebar():
    # history 초기화
    if "history" not in st.session_state:
        st.session_state.history = { label: [] for label in LABELS }

    st.sidebar.empty()
    with st.sidebar:
        render_lock_icon(size_px=100, color="#2E86AB")

        # 1) 카테고리 선택
        selected_label = st.selectbox(
            "📌 분류 사례",
            options=LABELS,
            key="selected_label"
        )
         # 2) 선택한 카테고리 간단 설명
        st.markdown("**설명**")
        st.write(CATEGORY_DESC[selected_label])
        st.markdown("---")

        # 2) 해당 카테고리의 저장된 Q&A 리스트
        entries = st.session_state.history[selected_label]

        if entries:
            st.markdown(f"### {selected_label}에 저장된 질문들")

            # 3) 질문만 뽑아 selectbox
            qs = [e["question"] for e in entries]
            selected_q = st.selectbox(
                "📝 저장된 질문",
                options=qs,
                key="selected_question"
            )

            # 4) 질문 불러오기 버튼: user_input에 세팅 후 rerun
            if st.button("질문 불러오기", key="btn_load"):
                st.session_state.user_input = selected_q
                st.rerun()
        else:
            st.write("아직 기록된 질문이 없습니다.")


def render_lock_icon(size_px=80, color="#444"):
    svg = f"""
    <div style="text-align:center; margin-bottom:1rem;">
              <svg xmlns="http://www.w3.org/2000/svg"
                   width="96" height="96"
                   viewBox="0 0 24 24" fill="#0B7ABF">
                <path d="M12 17a2 2 0 1 0-.002-3.998A2 2 0 0 0 12 17zm6-7h-1V7
                         c0-2.757-2.243-5-5-5S7 4.243 7 7v3H6
                         c-1.103 0-2 .897-2 2v9a2 2 0 0 0 2 2h12
                         a2 2 0 0 0 2-2v-9c0-1.103-.897-2-2-2zm-9-3
                         c0-1.654 1.346-3 3-3s3 1.346 3 3v3H9V7zm9 14
                         H6v-9h12v9z"/>
              </svg>
            </div>
    """
    st.markdown(svg, unsafe_allow_html=True)
