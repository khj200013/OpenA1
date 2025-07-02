import streamlit as st
from pathlib import Path
import base64

image_base_path = Path(__file__).parent.parent / "static" 
user_icon = image_base_path / 'user_icon_image.png'
assist_icon = image_base_path / 'assistant_icon_image.png'
title_img = image_base_path / 'logo_01.png'

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
                    full_response += content
                    response_container.markdown(full_response + "▍")
        except Exception as e:
            st.error(f"❌ 스트리밍 중 오류 발생: {e}")
            full_response = "응답을 받지 못했습니다."

    return full_response