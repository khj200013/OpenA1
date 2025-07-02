import streamlit as st

def render_user_input():
    return st.chat_input("개인정보 보호 관련 질문을 입력하세요")

def render_user_message(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)

def render_assistant_message(predicted_label, response):
    with st.chat_message("assistant"):
        st.markdown(f"🔍 예측된 분류: **{predicted_label}**")
        st.markdown(response)

def render_assistant_message_stream(predicted_label, stream):
    with st.chat_message("assistant"):
        st.markdown(f"🔍 예측된 분류: **{predicted_label}**")
        response_text = st.write_stream(stream)
    return response_text
