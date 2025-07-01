import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_metric
from sklearn.metrics import confusion_matrix, classification_report

# 1. 데이터 로드 (privacy_legal_data.csv 사용)
data = [
    ("회사에서 내 얼굴 사진을 허락 없이 홍보에 사용했어요",0),
    ("내 사진이 무단으로 SNS에 올라갔어요",0),
    ("홍보용 포스터에 제 얼굴이 동의 없이 사용되었어요",0),
    ("동료가 제 사진을 몰래 찍어서 인터넷에 올렸습니다",0),
    ("지하철에서 몰래 사진 찍혔습니다",0),
    ("학원에서 제 사진을 허락 없이 웹사이트에 게시했어요",0),
    ("아무런 동의 없이 영상 촬영을 당했어요",0),
    ("내 얼굴이 광고에 사용됐는데 사전 동의가 없었어요",0),
    ("학교 행사 사진에 제 얼굴이 나왔는데 동의한 적 없습니다",0),
    ("카페에서 몰래 사진 찍혀서 기분이 나쁩니다",0),

    ("내 개인정보가 유출된 것 같아요",1),
    ("스팸 전화가 너무 자주 와서 정보가 새나간 듯합니다",1),
    ("회사에서 수집한 정보가 외부에 공개됐어요",1),
    ("내 주소가 인터넷에 올라가 있어요",1),
    ("이메일 주소가 유출돼서 스팸이 쏟아집니다",1),
    ("누군가 내 주민등록번호를 알고 있어요",1),
    ("개인정보가 유출돼 금융사기로 이어졌어요",1),
    ("어디선가 내 휴대폰 번호를 알아냈어요",1),
    ("회사 내부 직원이 정보를 외부로 유출했습니다",1),
    ("정보가 해킹으로 빠져나간 것 같습니다",1),

    ("동의 없이 개인정보를 수집했어요",2),
    ("사전 고지 없이 개인정보 동의를 받지 않았어요",2),
    ("동의서도 없이 제 정보를 입력받았어요",2),
    ("회원가입 시 개인정보 동의 절차가 없었습니다",2),
    ("내가 동의하지 않은 마케팅 정보가 문자로 옵니다",2),
    ("동의하지 않은 제 정보를 사용하고 있어요",2),
    ("앱이 내 정보를 수집하는데 동의한 적이 없어요",2),
    ("사이트에서 동의 없이 내 위치정보를 추적합니다",2),
    ("내 기록을 동의 없이 분석하고 있어요",2),
    ("처리 목적이 명확하지 않은데도 동의를 요구하지 않았어요",2),

    ("동의 없이 CCTV 영상이 수집되었어요",2),
    ("마케팅 정보 수신 거부했는데 계속 연락이 와요",2),
    ("앱을 설치하자마자 위치정보를 가져갔어요",2),
    ("공공기관에서 개인정보 수집 시 고지가 없었습니다",2),
    ("개인정보 활용 동의 창이 아예 없었어요",2),
    ("회사에서 내 동의 없이 제 정보를 파트너사에 제공했어요",2),
    ("정보 제공 동의 절차가 투명하지 않았습니다",2),
    ("내 정보가 마케팅에 활용되고 있는데 동의한 기억이 없어요",2),
    ("동의 없이 내 정보를 제3자에게 넘긴 것 같아요",2),
    ("내가 제공한 적 없는 정보가 시스템에 등록돼 있어요",2),
]

df = pd.DataFrame(data, columns=["text", "label"])
label_map = {0: "초상권 침해", 1: "개인정보 유출", 2: "동의 미이행"}

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

# 3. KLUE BERT 모델 불러오기
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

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
    report_to="none"  # wandb 끄기
)

# 7. Trainer 객체
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 8. 학습 시작
trainer.train()

model.eval()
all_preds = []
all_labels = []

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

# 혼동 행렬 출력
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# 상세 분류 보고서 출력
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[label_map[i] for i in range(3)]))

# ✅ 10. 모델 저장 (모델 + 토크나이저)
model.save_pretrained("./saved_kobert_model")
tokenizer.save_pretrained("./saved_kobert_model")

# 9. 추론 함수
def classify_legal_issue(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

# 10. 예측 테스트
test_sentence = "회사에서 내 얼굴 사진을 허락 없이 홍보에 썼음"
print(f"입력: {test_sentence}")
print(f"예측된 유형: {classify_legal_issue(test_sentence)}")
