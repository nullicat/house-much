# ============================================================
# STEP 7: 피처 엔지니어링 & 전처리 (수정 버전)
# ============================================================

print("\n" + "=" * 80)
print("🔧 STEP 7: 피처 엔지니어링 & 전처리")
print("=" * 80)

import pandas as pd
import numpy as np

# ============================================================
# 7-1. 파생 변수 생성
# ============================================================

print("\n[7-1] 파생 변수 생성")

df_featured = df_sold.copy()

# 기본 파생 변수
print("\n   ① 기본 비율 변수")
df_featured['최저가율'] = df_featured['최저가'] / df_featured['감정가']
df_featured['보증금비율'] = df_featured['보증금'] / df_featured['감정가']
df_featured['토지건물비율'] = df_featured['건물면적'] / (df_featured['토지면적'] + 1)  # 0 방지
df_featured['평당감정가'] = df_featured['감정가'] / (df_featured['건물면적'] + 1)

print(f"      - 최저가율: {df_featured['최저가율'].mean():.3f}")
print(f"      - 보증금비율: {df_featured['보증금비율'].mean():.3f}")
print(f"      - 토지건물비율: {df_featured['토지건물비율'].mean():.3f}")
print(f"      - 평당감정가: {df_featured['평당감정가'].mean():,.0f}원")

# 이진 변수
print("\n   ② 이진 변수")
df_featured['보증금유무'] = (df_featured['보증금'].notna() & (df_featured['보증금'] > 0)).astype(int)
df_featured['선순위초과'] = (df_featured['보증금'] > df_featured['감정가']).astype(int)
df_featured['신건여부'] = (df_featured['유찰횟수'] == 0).astype(int)

print(f"      - 보증금 있음: {df_featured['보증금유무'].sum():,}개 ({df_featured['보증금유무'].mean()*100:.1f}%)")
print(f"      - 선순위 초과: {df_featured['선순위초과'].sum():,}개 ({df_featured['선순위초과'].mean()*100:.1f}%)")
print(f"      - 신건: {df_featured['신건여부'].sum():,}개 ({df_featured['신건여부'].mean()*100:.1f}%)")

# 날짜 파생 변수 (수정!)
print("\n   ③ 날짜 파생 변수")
# 혼합 형식 처리
df_featured['매각일'] = pd.to_datetime(df_featured['매각일'], format='mixed', errors='coerce')
df_featured['매각_연도'] = df_featured['매각일'].dt.year
df_featured['매각_월'] = df_featured['매각일'].dt.month
df_featured['매각_분기'] = df_featured['매각일'].dt.quarter

print(f"      - 매각 연도: {df_featured['매각_연도'].min():.0f} ~ {df_featured['매각_연도'].max():.0f}")
print(f"      - 매각 월: {df_featured['매각_월'].min():.0f} ~ {df_featured['매각_월'].max():.0f}")

# ============================================================
# 7-2. 결측치 처리
# ============================================================

print("\n[7-2] 결측치 처리")

# 보증금 관련 (NaN → 0)
df_featured['보증금'] = df_featured['보증금'].fillna(0)
df_featured['보증금비율'] = df_featured['보증금비율'].fillna(0)

# 토지/건물면적 (소수 결측 → 중앙값)
df_featured['토지면적'] = df_featured['토지면적'].fillna(df_featured['토지면적'].median())
df_featured['건물면적'] = df_featured['건물면적'].fillna(df_featured['건물면적'].median())

# 날짜 결측 확인
if df_featured['매각일'].isna().sum() > 0:
    print(f"   ⚠️ 매각일 결측: {df_featured['매각일'].isna().sum()}개")
    # 결측 행 제거
    df_featured = df_featured[df_featured['매각일'].notna()].copy()
    print(f"   → 제거 후: {len(df_featured):,}개")

# 기타 결측 확인
missing_after = df_featured.isnull().sum()
missing_after = missing_after[missing_after > 0]

if len(missing_after) > 0:
    print("\n   남은 결측치:")
    for col, count in missing_after.items():
        print(f"      {col}: {count}개")
else:
    print("\n   ✅ 모든 결측치 처리 완료!")

# ============================================================
# 7-3. Train/Test 분할 (시간 기준)
# ============================================================

print("\n[7-3] Train/Test 분할 (시간 기준)")

# 2021-2024: Train, 2025: Test
df_train = df_featured[df_featured['연도'] < 2025].copy()
df_test = df_featured[df_featured['연도'] == 2025].copy()

print(f"\n   Train (2021-2024): {len(df_train):,}개 ({len(df_train)/len(df_featured)*100:.1f}%)")
print(f"   Test (2025):       {len(df_test):,}개 ({len(df_test)/len(df_featured)*100:.1f}%)")

# 연도별 분포 확인
print("\n   Train 연도별 분포:")
print(df_train['연도'].value_counts().sort_index())

# ============================================================
# 7-4. 범주형 변수 인코딩 (Target Encoding)
# ============================================================

print("\n[7-4] 범주형 변수 인코딩")

# 용도: One-Hot (7개로 적음)
print("\n   ① 용도 → One-Hot Encoding")
용도_dummies_train = pd.get_dummies(df_train['용도'], prefix='용도')
용도_dummies_test = pd.get_dummies(df_test['용도'], prefix='용도')

# Train에 있는 모든 컬럼이 Test에도 있도록
for col in 용도_dummies_train.columns:
    if col not in 용도_dummies_test.columns:
        용도_dummies_test[col] = 0

print(f"      생성된 컬럼: {len(용도_dummies_train.columns)}개")

# 구/동: Target Encoding (고카디널리티)
print("\n   ② 구 → Target Encoding")

# Train에서만 평균 계산
구별_평균 = df_train.groupby('구')['낙찰가율'].mean()
전체_평균 = df_train['낙찰가율'].mean()

# Train 적용
df_train['구_encoded'] = df_train['구'].map(구별_평균)

# Test 적용 (Train에 없는 구는 전체 평균)
df_test['구_encoded'] = df_test['구'].map(구별_평균).fillna(전체_평균)

print(f"      Train 고유값: {df_train['구'].nunique()}개")
print(f"      Test 고유값: {df_test['구'].nunique()}개")
print(f"      인코딩 범위: {df_train['구_encoded'].min():.3f} ~ {df_train['구_encoded'].max():.3f}")

print("\n   ③ 동 → Target Encoding")

# Train에서만 평균 계산
동별_평균 = df_train.groupby('동')['낙찰가율'].mean()

# Train 적용
df_train['동_encoded'] = df_train['동'].map(동별_평균)

# Test 적용 (Train에 없는 동은 같은 구의 평균)
df_test['동_encoded'] = df_test['동'].map(동별_평균)

# Test에서 Train에 없는 동 처리
test_missing = df_test['동_encoded'].isna()
if test_missing.sum() > 0:
    print(f"      Test에만 있는 동: {test_missing.sum()}개 → 같은 구의 평균으로 대체")
    # 같은 구의 평균으로 대체
    for idx in df_test[test_missing].index:
        구 = df_test.loc[idx, '구']
        df_test.loc[idx, '동_encoded'] = 구별_평균.get(구, 전체_평균)

print(f"      Train 고유값: {df_train['동'].nunique()}개")
print(f"      Test 고유값: {df_test['동'].nunique()}개")
print(f"      인코딩 범위: {df_train['동_encoded'].min():.3f} ~ {df_train['동_encoded'].max():.3f}")

# 용도 One-Hot 결합
df_train = pd.concat([df_train, 용도_dummies_train], axis=1)
df_test = pd.concat([df_test, 용도_dummies_test], axis=1)

# ============================================================
# 7-5. 피처 선택
# ============================================================

print("\n[7-5] 피처 선택")

# 사용할 피처 리스트
feature_cols = [
    # 원본 수치
    '층', '토지면적', '건물면적', '감정가', '최저가', '유찰횟수',

    # 파생 비율
    '최저가율', '보증금비율', '토지건물비율', '평당감정가',

    # 이진 변수
    '보증금유무', '선순위초과', '신건여부',

    # 날짜
    '매각_월', '매각_분기',

    # 인코딩
    '구_encoded', '동_encoded',
]

# 용도 One-Hot 컬럼 추가
용도_cols = [col for col in df_train.columns if col.startswith('용도_')]
feature_cols.extend(용도_cols)

print(f"\n   총 피처 개수: {len(feature_cols)}개")

# X, y 분할
X_train = df_train[feature_cols].copy()
y_train = df_train['낙찰가율'].copy()

X_test = df_test[feature_cols].copy()
y_test = df_test['낙찰가율'].copy()

print(f"\n   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   X_test shape:  {X_test.shape}")
print(f"   y_test shape:  {y_test.shape}")

# 결측 최종 확인
print(f"\n   X_train 결측: {X_train.isnull().sum().sum()}개")
print(f"   X_test 결측: {X_test.isnull().sum().sum()}개")

# ============================================================
# 최종 확인
# ============================================================

print("\n" + "=" * 80)
print("✅ 피처 엔지니어링 & 전처리 완료!")
print("=" * 80)

print(f"""
📊 최종 데이터:
   - Train: {len(X_train):,}개 (2021-2024년)
   - Test:  {len(X_test):,}개 (2025년)
   - 피처: {len(feature_cols)}개
   - 타겟: 낙찰가율 (평균 {y_train.mean():.3f})

🎯 다음 단계:
   1. 베이스라인 모델 (Naive Mean)
   2. 선형 모델 (Ridge)
   3. 통계 검정 (H1: Model vs Baseline)
""")

print("=" * 80)