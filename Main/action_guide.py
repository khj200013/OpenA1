import json
from typing import Dict, List, Union
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def action_guide_agent(category: str, user_question: str) -> Dict[str, Union[str, List[str]]]:
    
    prompt = f"""
        당신은 개인정보 보호 및 디지털 권리 침해에 대한 법률 상담 전문가입니다.
        아래는 사용자 질문과 예측된 법률 카테고리입니다.
        이 정보에 기반하여 '신고 및 대응 절차'를 구체적이고 단계적으로 작성해주세요.

        [예측된 법률 카테고리]
        {category}

        [사용자 질문]
        {user_question}

        요구사항:
        - 신고 방법, 기관 이름, 소송 등 실질적인 대응 조치를 단계별로 안내하세요.
        - 관련 기관 웹사이트나 전화번호가 있다면 포함하세요.
        - 사용자가 실제 행동할 수 있도록 구체적으로 작성하세요.
        """

    response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "당신은 법률 상담 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
    
    return response
