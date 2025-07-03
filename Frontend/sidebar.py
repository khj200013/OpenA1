import streamlit as st
from pathlib import Path

LABELS = [
    "초상권 침해",
    "동의 없는 개인정보 수집",
    "목적 외 이용",
    "제3자 무단 제공",
    "CCTV 과잉촬영",
    "정보 유출",
    "파기 미이행",
    "광고성 정보 수신",
    "계정/비밀번호 관련 문제",
    "위치정보 수집/유출",
    "개인정보 열람·정정 요구 거부",
]

def render_history_sidebar():
    # 히스토리가 없으면 초기화
    if "history" not in st.session_state:
        st.session_state.history = { label:[] for label in LABELS }

    st.sidebar.empty()
    with st.sidebar:
        render_lock_icon(size_px=100, color="#2E86AB")
        # 1) 카테고리 선택박스
        selected = st.selectbox(
            "📌 분류 사례",
            options=LABELS,
            key="selected_label"
        )
        st.markdown("---")

        # 2) 해당 카테고리 히스토리 로드
        q_list = st.session_state.history.get(selected, [])

        # 3) 출력 로직
        if q_list:
            st.markdown(f"### {selected}에 해당하는 질문들")
            for i, q in enumerate(q_list, 1):
                st.write(f"{i}. {q}")
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