# 설치 필요 시 (주석 해제)
# !pip install transformers datasets

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_metric
from sklearn.metrics import confusion_matrix, classification_report

# 1. 9개 라벨 정의 및 예시 데이터 구성
label_map = {
    0: "초상권 침해",
    1: "동의 없는 개인정보 수집",
    2: "목적 외 이용",
    3: "제3자 무단 제공",
    4: "CCTV 과잉촬영",
    5: "정보 유출",
    6: "파기 미이행",
    7: "광고성 정보 수신",
    8: "계정/비밀번호 관련 문제"
}

data = [
    # 0: 초상권 침해
    ("회사 홈페이지에 제 사진이 무단으로 게재됐어요", 0),
    ("행사 중 찍힌 제 얼굴 사진이 광고에 사용됐습니다", 0),
    ("허락 없이 사진이 SNS에 올려졌어요", 0),
    ("촬영 동의 없이 뉴스에 얼굴이 나왔어요", 0),
    ("상사 몰래 촬영된 제 사진이 전사 메일로 배포됐습니다", 0),
    ("카페에서 누군가 제 사진을 찍어 올렸어요", 0),
    ("길거리 인터뷰 후 초상권 동의 없이 방송에 나왔어요", 0),
    ("포스터에 제 얼굴이 실렸는데 아무 동의한 적 없어요", 0),
    ("제 사진이 블로그에 무단으로 올라와 있었습니다", 0),
    ("친구가 동의 없이 사진을 단체방에 공유했어요", 0),
    ("강의 중 찍힌 제 모습이 강의 홍보 영상에 나왔어요", 0),
    ("연예인도 아닌데 제 얼굴이 온라인 기사에 나왔어요", 0),
    ("모임 사진에서 제 얼굴만 클로즈업해서 게시했어요", 0),
    ("몰래 찍은 사진이 인스타에 태그되어 있었어요", 0),
    ("지하철에서 누군가 몰래 사진을 찍었습니다", 0),
    ("운동 중 찍힌 영상이 동의 없이 유튜브에 올라갔어요", 0),
    ("회사 행사 사진이 제 얼굴을 강조해서 써요", 0),
    ("강제 촬영 후 삭제 요청을 무시했어요", 0),
    ("사진 촬영 거절했는데 끝까지 찍었어요", 0),
    ("모임 리더가 제 사진을 홍보용으로 써버렸어요", 0),

    # 1: 동의 없는 개인정보 수집
    ("회원가입 시 동의 없이 주민번호를 수집했어요", 1),
    ("동의 없이 위치정보를 가져갔어요", 1),
    ("앱을 설치했더니 정보 수집 동의가 자동으로 되어 있었어요", 1),
    ("이메일 수집에 대해 고지받지 못했어요", 1),
    ("문의했더니 제 개인정보를 이미 가지고 있었어요", 1),
    ("이용 동의서 없이 개인정보를 입력해야 했어요", 1),
    ("로그인할 때 동의 받지 않고 추가 정보를 요구해요", 1),
    ("앱이 몰래 마이크 권한을 가져갔어요", 1),
    ("고객센터에서 이름과 주소를 자동 저장해요", 1),
    ("신청서에 불필요한 정보까지 요구해요", 1),
    ("광고 클릭했더니 자동으로 정보가 넘어갔어요", 1),
    ("카페 회원가입에 생년월일을 무조건 입력해야 해요", 1),
    ("비밀번호 없이 제 계정 정보를 볼 수 있었어요", 1),
    ("설문조사에 과도한 개인정보 입력을 요구해요", 1),
    ("앱 설치 시 약관 동의 전에 위치정보가 수집되었어요", 1),
    ("웹사이트가 자동으로 제 정보를 긁어갔어요", 1),
    ("전화 상담 중 제 정보를 기록하는 것 같아요", 1),
    ("블로그 댓글에 이메일 입력이 강제돼요", 1),
    ("카카오 연동 시 과도한 정보가 수집돼요", 1),
    ("동의 없이 통화내용을 수집한 것 같아요", 1),

    # 2: 목적 외 이용
    ("마케팅에 내 정보를 사용하는 것 같아요", 2),
    ("이벤트 참여했더니 스팸이 오고 있어요", 2),
    ("약관에는 없던 용도로 제 정보를 사용하고 있어요", 2),
    ("본래 목적과 무관한 광고에 제 정보가 쓰였습니다", 2),
    ("설문조사 참여 후 광고성 연락이 와요", 2),
    ("비회원 주문했는데 연락이 계속 옵니다", 2),
    ("문의만 했는데 광고 문자가 와요", 2),
    ("이벤트 신청 정보가 광고에 활용됐어요", 2),
    ("연락처를 마케팅에 쓴다고 들은 적 없어요", 2),
    ("목적을 설명하지 않고 정보를 수집했어요", 2),
    ("의료 정보가 보험 상품 추천에 사용됐습니다", 2),
    ("물품 구매 시 수집한 정보를 다른 용도로 사용 중이에요", 2),
    ("문의했던 업체 외에 다른 데서 전화가 왔어요", 2),
    ("내 번호를 다른 목적에 활용 중인 것 같아요", 2),
    ("체험단 신청 후 스팸이 계속 옵니다", 2),
    ("내 동의 없이 다른 홍보에 이름이 사용됐어요", 2),
    ("내 정보가 타사 분석에 쓰였어요", 2),
    ("단순 문의였는데 정보가 재사용된 느낌이에요", 2),
    ("내 주소가 광고 배달에 쓰이고 있어요", 2),
    ("방문기록을 마케팅에 쓴다고 적혀 있었나요?", 2),

    # 3: 제3자 무단 제공
    ("내 개인정보가 제3자에게 동의 없이 넘어간 것 같아요", 3),
    ("다른 업체에서 내 정보를 활용하고 있어요", 3),
    ("서비스 가입 후 타사에서 연락이 옵니다", 3),
    ("내 정보가 광고 회사에 전달된 것 같습니다", 3),
    ("어느 순간 모르는 곳에서 내 정보가 사용됐어요", 3),
    ("동의하지 않았는데 제휴사에서 문자가 왔어요", 3),
    ("개인정보가 협력업체에 넘겨졌습니다", 3),
    ("타사에서 내 이메일을 알고 있더라고요", 3),
    ("제3자에게 개인정보를 제공한다고 들은 적이 없어요", 3),
    ("갑자기 여러 쇼핑몰에서 내 정보를 사용하고 있어요", 3),
    ("동의 없이 다른 기관에 정보가 넘어갔어요", 3),
    ("내 정보가 파트너 기업에 공유된 걸 알게 됐어요", 3),
    ("계약 당시 고지되지 않은 곳에서 연락이 왔어요", 3),
    ("광고 대행사가 내 번호를 가지고 있더군요", 3),
    ("타업체의 이벤트에서 내 정보가 활용되었어요", 3),
    ("동의 안 했는데 보험사에서 전화가 옵니다", 3),
    ("기입한 정보가 제3자와 공유된 정황이 있어요", 3),
    ("고객센터가 아닌 다른 곳에서 내 정보를 언급했어요", 3),
    ("내 정보가 외부 컨설팅 업체에 넘어간 것 같습니다", 3),
    ("공공기관 제출 정보가 민간업체에 제공됐어요", 3),

    # 4: CCTV 과잉촬영
    ("지하철 출입구에 CCTV가 너무 많아 사생활 침해가 우려됩니다", 4),
    ("공용 화장실 입구에 CCTV가 설치되어 있습니다", 4),
    ("헬스장 탈의실 근처에 CCTV가 있는 것 같아요", 4),
    ("회사 휴게실에 CCTV가 있어서 불편합니다", 4),
    ("학교 복도에 너무 많은 CCTV가 있어 감시받는 느낌이에요", 4),
    ("엘리베이터 내부까지 CCTV가 촬영하고 있습니다", 4),
    ("CCTV가 책상 바로 위에 있어 사생활이 침해돼요", 4),
    ("CCTV 안내 문구 없이 촬영되고 있었습니다", 4),
    ("아파트 주차장에 CCTV가 너무 가까이 설치돼 있어요", 4),
    ("회의실 내부에 CCTV가 작동 중이에요", 4),
    ("감시 목적 없이 설치된 CCTV가 너무 많아요", 4),
    ("어린이집 내부에 CCTV가 무분별하게 설치돼 있어요", 4),
    ("직원 휴게 공간에 CCTV가 몰래 설치되어 있었어요", 4),
    ("개인상담실에 CCTV가 있다는 사실을 몰랐어요", 4),
    ("카페 내부를 상시 촬영하는 CCTV가 불편해요", 4),
    ("시설 내 사전 고지 없이 설치된 CCTV가 있어요", 4),
    ("학생 자습 공간에 CCTV가 상시 작동 중입니다", 4),
    ("민감한 공간까지 CCTV가 촬영하고 있습니다", 4),
    ("상시 촬영 여부에 대한 고지가 없어요", 4),
    ("사무실 내 CCTV 영상이 너무 자주 확인됩니다", 4),

    # 5: 정보 유출
    ("회사 서버 해킹으로 고객 정보가 유출되었어요", 5),
    ("이메일 주소가 외부에 퍼졌습니다", 5),
    ("내 정보가 인터넷에 돌아다녀요", 5),
    ("이름과 전화번호가 검색됩니다", 5),
    ("SNS에 개인 정보가 유출됐습니다", 5),
    ("금융 정보가 유출되어 피해를 입었습니다", 5),
    ("주민등록번호가 유출된 것 같아요", 5),
    ("개인정보가 담긴 문서를 분실했어요", 5),
    ("기록이 외부에 노출된 걸 발견했어요", 5),
    ("이용 중이던 앱에서 정보가 유출됐어요", 5),
    ("개인 사진이 유출되었어요", 5),
    ("메일 주소가 해커에게 넘어간 것 같아요", 5),
    ("내 아이디가 외부에 노출됐습니다", 5),
    ("고객 정보가 유출됐다는 통지를 받았어요", 5),
    ("모르는 사람이 내 정보를 언급했어요", 5),
    ("회사 내부 실수로 정보가 누출됐어요", 5),
    ("사용자 로그가 외부에서 확인 가능했습니다", 5),
    ("내 정보가 검색사이트에 올라와 있어요", 5),
    ("기기에 저장된 정보가 유출된 듯해요", 5),
    ("정보 유출로 인한 금전적 피해를 입었습니다", 5),

    # 6: 파기 미이행
    ("탈퇴했는데 개인정보가 아직도 남아 있어요", 6),
    ("계약 종료 후에도 정보가 삭제되지 않았어요", 6),
    ("보관 기한이 지났는데도 파기하지 않았어요", 6),
    ("정보 삭제 요청을 무시당했어요", 6),
    ("개인 정보가 여전히 시스템에 남아있어요", 6),
    ("이미 파기했어야 할 정보가 보관 중이에요", 6),
    ("회사에서 파기 의무를 지키지 않았어요", 6),
    ("동의 철회 후에도 정보가 유지됐어요", 6),
    ("고객 DB에서 삭제되지 않았어요", 6),
    ("개인 정보가 계속 활용되고 있어요", 6),
    ("내 기록이 서버에 남아 있었어요", 6),
    ("파기 요청했는데 답변이 없어요", 6),
    ("문서 파기 절차가 없다고 합니다", 6),
    ("파일 삭제가 제대로 되지 않았어요", 6),
    ("메일 탈퇴했는데 정보는 남아있어요", 6),
    ("계약 만료 후에도 DB에 남겨뒀어요", 6),
    ("파기했다고 했는데 확인해보니 있었어요", 6),
    ("사용자 데이터를 영구 보관하고 있어요", 6),
    ("이메일 삭제했는데 여전히 광고가 와요", 6),
    ("파일 백업본까지 지워달라고 했는데 남아있습니다", 6),

    # 7: 광고성 정보 수신
    ("수신 거부했는데 계속 스팸이 와요", 7),
    ("광고 문자가 멈추지 않습니다", 7),
    ("이메일 스팸이 하루에 수십 통이에요", 7),
    ("마케팅 동의 안 했는데 전화 옵니다", 7),
    ("광고 전화로 업무에 방해돼요", 7),
    ("앱에서 광고 알림을 꺼도 계속 와요", 7),
    ("문자 차단했는데도 다른 번호로 옵니다", 7),
    ("스팸 전화가 너무 자주 와요", 7),
    ("동의 없이 이벤트 안내가 와요", 7),
    ("수신 동의 없이 광고 메일을 받았어요", 7),
    ("수신 거부했는데도 다른 브랜드에서 연락와요", 7),
    ("메일 해지했는데 아직 옵니다", 7),
    ("광고 알림이 너무 잦아요", 7),
    ("마케팅 동의 철회했는데도 여전합니다", 7),
    ("스팸 차단이 무용지물이에요", 7),
    ("이벤트 참여 후 광고가 계속 와요", 7),
    ("회원가입 후 마케팅 동의한 적 없어요", 7),
    ("이용약관에 동의했을 뿐인데 광고가 옵니다", 7),
    ("광고 수신 동의 없이 전화가 와요", 7),
    ("동의 철회 후에도 광고 메시지가 왔어요", 7),

    # 8: 계정/비밀번호 관련 문제
    ("비밀번호를 변경했는데도 해킹당했어요", 8),
    ("계정에 누군가 무단 로그인했습니다", 8),
    ("비밀번호 재설정 메일이 자주 와요", 8),
    ("모르는 기기에서 접속이 감지됐습니다", 8),
    ("비밀번호를 바꿨는데 접속이 안 돼요", 8),
    ("계정 도용 의심됩니다", 8),
    ("이중 인증이 작동하지 않아요", 8),
    ("계정에 저장된 정보가 변경됐어요", 8),
    ("내가 로그인하지 않았는데 접속 기록이 있어요", 8),
    ("비밀번호 찾기 시도 기록이 남아있어요", 8),
    ("계정 복구가 안 됩니다", 8),
    ("계정이 정지됐는데 이유를 모르겠어요", 8),
    ("계정 해킹으로 피해를 입었어요", 8),
    ("계정 잠금이 자주 발생해요", 8),
    ("비밀번호가 자동으로 바뀌었습니다", 8),
    ("계정 접근 알림이 자주 와요", 8),
    ("내 계정이 다른 이메일과 연결됐어요", 8),
    ("누군가 내 계정을 사용했어요", 8),
    ("비밀번호 오류로 접속이 제한돼요", 8),
    ("비밀번호가 외부에 유출된 것 같아요", 8)

]

df = pd.DataFrame(data, columns=["text", "label"])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 2. 데이터셋 클래스 정의
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }

# 3. 모델과 토크나이저 로딩
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=9)

# 4. 데이터셋 준비
train_dataset = LegalDataset(train_texts, train_labels, tokenizer)
val_dataset = LegalDataset(val_texts, val_labels, tokenizer)

# 5. 평가 지표 설정
accuracy_metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# 6. 훈련 설정
training_args = TrainingArguments(
    output_dir='./klue_bert_results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 7. 학습 시작
trainer.train()

# 8. 혼동 행렬 및 평가 리포트 출력
model.eval()
all_preds, all_labels = [], []
for batch in val_dataset:
    inputs = {
        'input_ids': batch['input_ids'].unsqueeze(0),
        'attention_mask': batch['attention_mask'].unsqueeze(0)
    }
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    all_preds.append(pred)
    all_labels.append(batch['labels'].item())

print("\n[Confusion Matrix]")
print(confusion_matrix(all_labels, all_preds))

print("\n[Classification Report]")
print(classification_report(
    all_labels, 
    all_preds, 
    labels=list(label_map.keys()),
    target_names=[label_map[i] for i in range(len(label_map))]
))

# 9. 모델 저장
model.save_pretrained("./saved_klue_bert")
tokenizer.save_pretrained("./saved_klue_bert")

# 10. 추론 함수
def classify_legal_issue(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

# 11. 테스트 예시
example = "앱을 설치했더니 동의 없이 위치정보를 수집했어요"
print(f"\n입력: {example}\n예측된 유형: {classify_legal_issue(example)}")
