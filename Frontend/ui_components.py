import streamlit as st
from pathlib import Path
import base64
import re

image_base_path = Path(__file__).parent.parent / "static" 
user_icon = image_base_path / 'user_icon_image.png'
assist_icon = image_base_path / 'assistant_icon_image.png'
title_img = image_base_path / 'logo_01.png'

# HTML 태그 치환
def clean_html_streaming(text):
    # 1. <strong> → **bold**
    text = re.sub(r'<\s*strong\s*>(.*?)<\s*/\s*strong\s*>', r'**\1**', text, flags=re.IGNORECASE)

    # 2. <em> → *italic*
    text = re.sub(r'<\s*em\s*>(.*?)<\s*/\s*em\s*>', r'*\1*', text, flags=re.IGNORECASE)

    # 3. <br> or <br/> → 줄바꿈
    text = re.sub(r'<\s*br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)

    # 4. <ul>, </ul> → 제거 (리스트만 남김)
    text = re.sub(r'</?\s*ul\s*>', '', text, flags=re.IGNORECASE)

    # 5. <li> → - 리스트 항목
    text = re.sub(r'<\s*li\s*>', '- ', text, flags=re.IGNORECASE)

    # 6. </li> → 줄바꿈
    text = re.sub(r'</\s*li\s*>', '\n', text, flags=re.IGNORECASE)

    # 7. 기타 모든 태그 제거
    text = re.sub(r'<[^>]+>', '', text)

    # 8. 중복 줄바꿈 정리
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text

def render_law_info(law_info, file_search_result):
    with st.expander("📘 예측된 법률 정보", expanded=True):
        st.markdown(f"**관련 법 조항:** {law_info['law']}")
        st.markdown(f"**요약 설명:** {law_info['summary']}")
        st.markdown(f"**위반 가능성:** {law_info['violation']}")

        if file_search_result:
            st.markdown("**📄 개인정보보호법 조항:**")
            st.markdown(
                f"""
                <div style="white-space: pre-wrap; word-wrap: break-word;
                            border: 1px solid #ddd;
                            border-radius: 0.5rem;
                            padding: 0.75rem;
                            background-color: #f9f9f9;
                            margin-top: 0.5rem;
                            margin-bottom: 0.5rem;">
                {file_search_result}
                </div>
                """,
                unsafe_allow_html=True
            )

def render_title_image():
    with open(title_img, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()

    # 로고 이미지 출력
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <img src="data:image/png;base64,{logo_data}" alt="logo" style="width: 400px; max-width: 80%;"/>
        </div>
    """, unsafe_allow_html=True)

def render_user_input():
    return st.chat_input("개인정보 보호 관련 질문을 입력하세요")

def render_user_message(prompt):
    with st.chat_message("user", avatar=user_icon):
        st.markdown(prompt)


def render_assistant_message(predicted_label, response):
    with st.chat_message("assistant", avatar=assist_icon):
        st.markdown(f"🔍 예측된 분류: **{predicted_label}**")
        st.markdown(response)

def render_assistant_message_stream(predicted_label, stream):
    full_response = ""
    with st.chat_message("assistant", avatar=assist_icon):
        st.markdown(f"🧠 **예측된 유형:** `{predicted_label}`")
        response_container = st.empty()

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content is not None:
                    normalized_content = clean_html_streaming(content)
                    full_response += normalized_content
                    response_container.markdown(full_response + "▍")
        except Exception as e:
            st.error(f"❌ 스트리밍 중 오류 발생: {e}")
            full_response = "응답을 받지 못했습니다."

    return full_response
