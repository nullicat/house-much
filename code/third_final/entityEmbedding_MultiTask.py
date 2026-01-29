# ============================================================
# 🚀 Entity Embedding + Multi-Task Learning (구/동 수정 버전)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 임포트
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 80)
print("🚀 Entity Embedding + Multi-Task 딥러닝 (구/동 수정 버전)")
print("=" * 80)
print(f"TensorFlow 버전: {tf.__version__}")

# ============================================================
# [0] Checkpoint 로드 + 구/동 인코딩 수정
# ============================================================

print("\n[0] Checkpoint 로드 + 구/동 인코딩")

import pickle

backup_dir = '/content/drive/MyDrive/auction_project_backup'

with open(f'{backup_dir}/checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# df_sold에서 원본 구/동 인코딩
df_sold = checkpoint['df_sold']

print(f"   df_sold: {df_sold.shape}")

# 구/동 인코딩 (원본에서)
le_gu = LabelEncoder()
le_dong = LabelEncoder()

df_sold['구_encoded_new'] = le_gu.fit_transform(df_sold['구'])
df_sold['동_encoded_new'] = le_dong.fit_transform(df_sold['동'])

print(f"   ✅ 인코딩 완료")
print(f"   구: {len(le_gu.classes_)}개")
print(f"   동: {len(le_dong.classes_)}개")

# Train/Test 데이터
X_train = checkpoint['X_train']
X_test = checkpoint['X_test']
y_train = checkpoint['y_train']
y_test = checkpoint['y_test']

# Train/Test 인덱스 추출
train_indices = X_train.index
test_indices = X_test.index

# 구/동 인코딩 추출
gu_train = df_sold.loc[train_indices, '구_encoded_new'].values
gu_test = df_sold.loc[test_indices, '구_encoded_new'].values

dong_train = df_sold.loc[train_indices, '동_encoded_new'].values
dong_test = df_sold.loc[test_indices, '동_encoded_new'].values

print(f"\n   구 범위 (train): {gu_train.min()} ~ {gu_train.max()}")
print(f"   동 범위 (train): {dong_train.min()} ~ {dong_train.max()}")

print(f"\n   구 샘플: {gu_train[:5]}")
print(f"   동 샘플: {dong_train[:5]}")

# 수치형 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✅ 스케일링 완료")

# Array 변환
y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

print(f"   ✅ 데이터 준비 완료")

# ============================================================
# [1] 데이터 준비
# ============================================================

print("\n[1] 데이터 준비")

print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# 수치형 피처 (15개)
numeric_features = ['층', '토지면적', '건물면적', '감정가', '최저가',
                    '유찰횟수', '최저가율', '보증금비율', '토지건물비율',
                    '평당감정가', '보증금유무', '선순위초과', '신건여부',
                    '매각_월', '매각_분기']

# 인덱스
numeric_indices = [list(X_train.columns).index(f) for f in numeric_features]

# 수치형 데이터
X_train_numeric = X_train_scaled[:, numeric_indices]
X_test_numeric = X_test_scaled[:, numeric_indices]

print(f"   수치형: {X_train_numeric.shape}")

# 범주형 데이터 (구/동 - 정수!)
X_train_gu = gu_train.reshape(-1, 1)
X_train_dong = dong_train.reshape(-1, 1)

X_test_gu = gu_test.reshape(-1, 1)
X_test_dong = dong_test.reshape(-1, 1)

print(f"   범주형 (구): {X_train_gu.shape}")
print(f"   범주형 (동): {X_train_dong.shape}")

# 분류 타겟
y_train_class = (y_train_array < 0.5).astype(int)
y_test_class = (y_test_array < 0.5).astype(int)

print(f"   유찰 케이스: {y_train_class.sum()}개 ({y_train_class.sum()/len(y_train_class)*100:.1f}%)")

# ============================================================
# [2] 모델 구조
# ============================================================

print("\n[2] 모델 구조 정의")

# 구, 동의 고유값 개수
n_gu = len(le_gu.classes_)
n_dong = len(le_dong.classes_)

print(f"   구: {n_gu}개")
print(f"   동: {n_dong}개")

# 임베딩 차원
embedding_dim_gu = 5
embedding_dim_dong = 10  # 동이 더 많으니 차원 높임

# Input Layers
input_numeric = layers.Input(shape=(len(numeric_features),), name='numeric_input')
input_gu = layers.Input(shape=(1,), name='gu_input')
input_dong = layers.Input(shape=(1,), name='dong_input')

# Embedding Layers
embedding_gu = layers.Embedding(
    input_dim=n_gu,
    output_dim=embedding_dim_gu,
    name='gu_embedding'
)(input_gu)
embedding_gu = layers.Flatten()(embedding_gu)

embedding_dong = layers.Embedding(
    input_dim=n_dong,
    output_dim=embedding_dim_dong,
    name='dong_embedding'
)(input_dong)
embedding_dong = layers.Flatten()(embedding_dong)

# Concatenate
concat = layers.Concatenate()([
    input_numeric,
    embedding_gu,
    embedding_dong
])

# Shared Layers (Body)
x = layers.Dense(256, activation='relu')(concat)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Multi-Task Heads
# Head 1: 분류 (유찰 여부)
classification_head = layers.Dense(32, activation='relu', name='class_dense')(x)
classification_output = layers.Dense(1, activation='sigmoid', name='classification')(classification_head)

# Head 2: 회귀 (낙찰가율)
regression_head = layers.Dense(32, activation='relu', name='reg_dense')(x)
regression_output = layers.Dense(1, name='regression')(regression_head)

# Model
model = keras.Model(
    inputs=[input_numeric, input_gu, input_dong],
    outputs=[classification_output, regression_output]
)

print("   ✅ 모델 구조 완성")
print(f"   파라미터: {model.count_params():,}개")

# ============================================================
# [3] 컴파일
# ============================================================

print("\n[3] 컴파일")

# Multi-task Loss (회귀 중시)
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss={
        'classification': 'binary_crossentropy',
        'regression': keras.losses.Huber(delta=1.35)
    },
    loss_weights={
        'classification': 0.1,  # 분류 가중치 낮춤
        'regression': 1.0       # 회귀 중시
    },
    metrics={
        'classification': 'accuracy',
        'regression': 'mae'
    }
)

print("   ✅ 컴파일 완료")

# ============================================================
# [4] 학습
# ============================================================

print("\n[4] 학습 시작 (약 2~3분 소요)")

history = model.fit(
    [X_train_numeric, X_train_gu, X_train_dong],
    [y_train_class, y_train_array],
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_regression_mae',
            patience=15,
            restore_best_weights=True,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_regression_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min'
        )
    ],
    verbose=1
)

print(f"\n   ✅ 학습 완료 ({len(history.history['loss'])} epochs)")

# ============================================================
# [5] 평가
# ============================================================

print("\n" + "=" * 80)
print("[5] 평가")
print("=" * 80)

# 예측
class_pred, reg_pred = model.predict(
    [X_test_numeric, X_test_gu, X_test_dong],
    verbose=0
)

class_pred = class_pred.flatten()
reg_pred = reg_pred.flatten()

# 전체 성능
mae_total = mean_absolute_error(y_test_array, reg_pred)
print(f"\n   전체 MAE: {mae_total:.4f}")

baseline_mae = 0.0715
improvement = (baseline_mae - mae_total) / baseline_mae * 100
print(f"   Baseline: {baseline_mae:.4f}")
print(f"   개선: {improvement:+.1f}%")

# 저가 구간
low_mask = (y_test_array < 0.5)
mae_low = mean_absolute_error(y_test_array[low_mask], reg_pred[low_mask])
baseline_low = 0.0805

print(f"\n   저가 MAE: {mae_low:.4f}")
print(f"   Baseline: {baseline_low:.4f}")
print(f"   개선: {(baseline_low - mae_low) / baseline_low * 100:+.1f}%")

# Within 5%p
abs_errors = np.abs(reg_pred - y_test_array)
within_5p_total = (abs_errors <= 0.05).sum() / len(abs_errors) * 100
within_5p_low = (abs_errors[low_mask] <= 0.05).sum() / low_mask.sum() * 100

print(f"\n   전체 Within 5%p: {within_5p_total:.1f}%")
print(f"   Baseline: 49.7%")
print(f"   개선: {within_5p_total - 49.7:+.1f}%p")

print(f"\n   저가 Within 5%p: {within_5p_low:.1f}%")
print(f"   Baseline: 25.7%")
print(f"   개선: {within_5p_low - 25.7:+.1f}%p")

# 분류 성능
class_pred_binary = (class_pred > 0.5).astype(int)
acc = accuracy_score(y_test_class, class_pred_binary)
f1 = f1_score(y_test_class, class_pred_binary)

print(f"\n   유찰 분류 정확도: {acc:.3f}")
print(f"   F1 Score: {f1:.3f}")

# ============================================================
# [6] 혼동행렬
# ============================================================

print("\n" + "=" * 80)
print("[6] 혼동행렬")
print("=" * 80)

def categorize_error(error):
    if error <= 0.03:
        return 'Excellent'
    elif error <= 0.05:
        return 'Good'
    elif error <= 0.10:
        return 'Fair'
    else:
        return 'Poor'

dl_categories = [categorize_error(e) for e in abs_errors]
dl_counts = Counter(dl_categories)

total = len(abs_errors)

print("\n[Entity Embedding + Multi-Task]")
print(f"   Excellent (≤3%p): {dl_counts['Excellent']/total*100:.1f}%")
print(f"   Good (3~5%p):     {dl_counts['Good']/total*100:.1f}%")
print(f"   Fair (5~10%p):    {dl_counts['Fair']/total*100:.1f}%")
print(f"   Poor (>10%p):     {dl_counts['Poor']/total*100:.1f}%")

print(f"\n   Within 10%p: {((dl_counts['Excellent'] + dl_counts['Good'] + dl_counts['Fair'])/total*100):.1f}%")

# ============================================================
# [7] 최종 요약
# ============================================================

print("\n" + "=" * 80)
print("🏆 최종 요약")
print("=" * 80)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                성능 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[전체 성능]
Baseline MAE:       {baseline_mae:.4f}
DL MAE:             {mae_total:.4f}
개선:               {improvement:+.1f}%

[저가 성능]
Baseline MAE:       {baseline_low:.4f}
DL MAE:             {mae_low:.4f}
개선:               {(baseline_low - mae_low) / baseline_low * 100:+.1f}%

[Within 5%p]
Baseline (전체):    49.7%
DL (전체):          {within_5p_total:.1f}% ({within_5p_total - 49.7:+.1f}%p)

Baseline (저가):    25.7%
DL (저가):          {within_5p_low:.1f}% ({within_5p_low - 25.7:+.1f}%p)

[혼동행렬]
Excellent + Good:   {(dl_counts['Excellent'] + dl_counts['Good'])/total*100:.1f}%
Fair:               {dl_counts['Fair']/total*100:.1f}%
Poor:               {dl_counts['Poor']/total*100:.1f}%

[분류 성능]
유찰 예측 정확도:   {acc:.1%}
F1 Score:           {f1:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 평가
if improvement > 5:
    print("🎉🎉🎉 대폭 개선! Entity Embedding + Multi-Task 효과 확인!")
elif improvement > 2:
    print("✅✅ 개선됨! 딥러닝 효과 있음!")
elif improvement > 0:
    print("✅ 소폭 개선")
else:
    print("⚠️ 개선 안 됨. 추가 튜닝 필요")

if within_5p_low > 35:
    print(f"🎉🎉🎉 저가 구간 대폭 개선! (+{within_5p_low - 25.7:.1f}%p)")
elif within_5p_low > 30:
    print(f"✅✅ 저가 구간 개선! (+{within_5p_low - 25.7:.1f}%p)")

if dl_counts['Poor']/total*100 < 12:
    print("🎉🎉🎉 Poor 구간 크게 감소! 목표 달성!")
elif dl_counts['Poor']/total*100 < 14:
    print("✅✅ Poor 구간 감소!")

print("\n" + "=" * 80)

# ============================================================
# [8] 결과 저장
# ============================================================

print("\n[8] 결과 저장")

# 예측값 저장
dl_predictions_fixed = {
    'y_test': y_test_array,
    'reg_pred': reg_pred,
    'class_pred': class_pred,
    'mae_total': mae_total,
    'mae_low': mae_low,
    'within_5p_total': within_5p_total,
    'within_5p_low': within_5p_low,
    'dl_counts': dl_counts
}

# Checkpoint에 추가
checkpoint['dl_predictions_fixed'] = dl_predictions_fixed
checkpoint['dl_history_fixed'] = history.history
checkpoint['le_gu'] = le_gu
checkpoint['le_dong'] = le_dong

# 저장
with open(f'{backup_dir}/checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

print(f"   ✅ 결과 저장 완료: checkpoint.pkl에 추가")
print("\n" + "=" * 80)