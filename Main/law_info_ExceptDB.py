import openai

# OpenAI API 키 설정
openai.api_key = "YOUR_OPENAI_API_KEY"  # 환경변수 처리 권장

def get_law_info(issue_type: str) -> dict:
    prompt = f"""
다음은 개인정보 관련 이슈 유형입니다: "{issue_type}"

이에 해당하는 법률 조항(개인정보보호법 또는 GDPR), 법적 해석, 위반 가능성,  
그리고 사용자가 참고할 수 있는 관련 링크를 아래 포맷에 맞춰 알려줘.

형식:
- 관련 법 조항: 
- 요약 설명: 
- 위반 가능성: 
- 참고 링크: (신뢰할 수 있는 법령 정보 또는 공식 기관)

비전문가도 쉽게 이해할 수 있게 써줘.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    answer = response['choices'][0]['message']['content']

    # 응답을 정리된 dict로 구성 (간단한 파싱)
    lines = answer.strip().split("\n")
    result = {
        "issue_type": issue_type,
        "law": lines[0].replace("- 관련 법 조항: ", "").strip(),
        "summary": lines[1].replace("- 요약 설명: ", "").strip(),
        "violation": lines[2].replace("- 위반 가능성: ", "").strip(),
        "reference": lines[3].replace("- 참고 링크: ", "").strip()
    }

    return result
