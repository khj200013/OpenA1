from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_law_info(issue_type: str, user_input: str) -> dict:
    prompt = f"""
다음은 개인정보 관련 이슈 유형입니다: "{issue_type}"

사용자가 말한 실제 상황:  
"{user_input}"

이에 해당하는 법률 조항(개인정보보호법 또는 GDPR), 법적 해석, 위반 가능성,  
그리고 사용자가 참고할 수 있는 관련 링크를 아래 포맷에 맞춰 알려줘.

형식:
- 관련 법 조항: 
- 요약 설명: 
- 위반 가능성: 
- 참고 링크: (신뢰할 수 있는 법령 정보 또는 공식 기관)

비전문가도 쉽게 이해할 수 있게 써줘.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
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
