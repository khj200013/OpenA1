from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_law_info(issue_type: str, user_input: str, law_article: str = None) -> dict:
 # 프롬프트 베이스
    prompt = f"""
다음은 개인정보 관련 이슈 유형입니다: "{issue_type}"

사용자 설명: "{user_input}"

"""
    if law_article:
        prompt += f'관련 법 조항: "{law_article}"\n\n'

    prompt += """
이에 맞는 법적 해석, 위반 가능성, 참고 링크를 아래 포맷에 맞춰 간결하게 알려줘.

- 관련 법 조항: 
- 요약 설명: 
- 위반 가능성: 
- 참고 링크: (공식 사이트)

비전문가도 쉽게 이해할 수 있게 작성.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )

        answer = response.choices[0].message.content.strip()
        lines = answer.split("\n")

        # 기본값 세팅
        result = {
            "issue_type": issue_type,
            "law": "",
            "summary": "",
            "violation": "",
            "reference": ""
        }

        # 응답 줄 개수만큼 파싱
        for line in lines:
            if line.startswith("- 관련 법 조항:"):
                result["law"] = line.replace("- 관련 법 조항:", "").strip()
            elif line.startswith("- 요약 설명:"):
                result["summary"] = line.replace("- 요약 설명:", "").strip()
            elif line.startswith("- 위반 가능성:"):
                result["violation"] = line.replace("- 위반 가능성:", "").strip()

        return result

    except Exception as e:
        return {
            "issue_type": issue_type,
            "law": "오류 발생",
            "summary": "GPT 응답 처리 중 문제가 발생했습니다.",
            "violation": str(e),
        }