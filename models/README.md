# 🤖 학습된 모델 파일

학습된 모델 파일은 용량 제한으로 깃허브에 포함되지 않습니다.

---

## 📥 **모델 다운로드**

### **구글 드라이브 링크**
[모델 다운로드 (구글 드라이브)](https://drive.google.com/drive/folders/1Bh5C93lrfTVpR_sExLlwCiKBcwT6zLwb).

---

## 📦 **필요한 모델 파일**

### **2단계 모델 (최종 선정)**

| 파일명 | 크기 | 설명 | 다운로드 |
|--------|------|------|----------|
| `2stage_classifier.pkl` | 13.7MB | Stage 1: 분류 모델 (정상 vs 유찰) | ⭐ 필수 |
| `2stage_huber_success.pkl` | 10KB | Stage 2-1: 정상 그룹 회귀 모델 | ⭐ 필수 |
| `2stage_huber_fail.pkl` | 2KB | Stage 2-2: 유찰 그룹 회귀 모델 | ⭐ 필수 |

### **기타 모델 (실험용)**

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `checkpoint.pkl` | 9.8MB | 중간 실험 결과 백업 (선택) |
| `bert_embeddings.pkl` | 47.1MB | BERT 임베딩 (NLP 실험용) |
| `pycaret_best_model.pkl` | 15KB | PyCaret AutoML 결과 |
| `catboost_model.cbm` | 659KB | CatBoost 모델 |
| `linear_model.pkl` | 788B | 선형 회귀 베이스라인 |

---

## 🚀 **모델 사용 방법**

### **1. 모델 파일 다운로드**

```bash
# 구글 드라이브에서 다운로드
# → models/ 폴더에 저장
cd models/
# 파일 복사:
# - 2stage_classifier.pkl
# - 2stage_huber_success.pkl
# - 2stage_huber_fail.pkl
```

### **2. Python에서 모델 로드**

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 모델 로드
clf = joblib.load('models/2stage_classifier.pkl')
huber_success = joblib.load('models/2stage_huber_success.pkl')
huber_fail = joblib.load('models/2stage_huber_fail.pkl')

# 데이터 준비 (예시)
X_new = np.array([[0.8]])  # 최저가율
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)

# Stage 1: 분류
group_pred = clf.predict(X_scaled)  # 0: 정상, 1: 유찰

# Stage 2: 회귀
if group_pred == 0:
    ratio_pred = huber_success.predict(X_scaled)[0]
else:
    ratio_pred = huber_fail.predict(X_scaled)[0]

print(f"예측 그룹: {'정상' if group_pred == 0 else '유찰'}")
print(f"예측 낙찰가율: {ratio_pred:.4f}")
```

---

## 🔄 **모델 재학습**

모델을 처음부터 재학습하려면:

```bash
# Jupyter 노트북 실행
jupyter notebook notebooks/3_모델링_최종.ipynb

# 또는 Python 스크립트
python src/model.py --train
```

---

## 📊 **모델 아키텍처**

### **Stage 1: RandomForest Classifier**
```
목적: 정상 그룹 vs 유찰 그룹 분류
알고리즘: RandomForestClassifier
하이퍼파라미터:
- n_estimators: 1000
- max_depth: 15
- class_weight: 'balanced'
- random_state: 42

성능 (2025년 검증):
- Accuracy: 97.0%
- Precision: 0.655
- Recall: 0.828
```

### **Stage 2-1: Huber Regressor (정상 그룹)**
```
목적: 정상 그룹 낙찰가율 예측
알고리즘: HuberRegressor
하이퍼파라미터:
- epsilon: 1.35
- alpha: 0.0001 (L2 정규화)
- max_iter: 100

성능 (2025년 검증):
- MAE: 0.0700
```

### **Stage 2-2: Huber Regressor (유찰 그룹)**
```
목적: 유찰 그룹 낙찰가율 예측
알고리즘: HuberRegressor
하이퍼파라미터:
- epsilon: 1.1 (정상 그룹과 다름!)
- alpha: 0.0001 (L2 정규화)
- max_iter: 100

성능 (2025년 검증):
- MAE: 0.0401 ⭐
```

---

## 📈 **모델 성능**

### **2025년 검증 결과**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 MAE: 0.0686 (6.86%p 오차)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
분류 정확도: 97.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

그룹별 성능:
- 정상 그룹 (≥0.5): MAE 0.0700
- 유찰 그룹 (<0.5): MAE 0.0401 ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 **문의**

모델 사용 중 문제가 발생하면 이슈를 등록해주세요.
- [GitHub Issues](https://github.com/your-username/seoul-auction-prediction/issues)

---

**⭐ 프로젝트가 도움이 되셨다면 Star를 눌러주세요! ⭐**
