# ============================================================
# 🏆 서울 경매 최종 분석 (Huber + SHAP + 앙상블)
# ============================================================

print("=" * 80)
print("🏆 서울 경매 최종 분석")
print("=" * 80)

print("""
프로젝트 요약:
- Baseline: MAE 0.1402
- CatBoost: MAE 0.0753 (46.3% 개선)
- Huber (PyCaret): MAE 0.0717 (48.9% 개선) ⭐

목표:
1. Huber SHAP 분석
2. Huber + CatBoost 앙상블
3. 최종 비교 및 시각화
""")

# ============================================================
# [1] 환경 설정
# ============================================================

print("\n[1] 환경 설정")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("   ✅ 라이브러리 로드 완료")

# ============================================================
# [2] 구글 드라이브 마운트
# ============================================================

print("\n[2] 구글 드라이브 마운트")

from google.colab import drive
drive.mount('/content/drive')

backup_dir = '/content/drive/MyDrive/auction_project_backup'
print(f"   경로: {backup_dir}")

# ============================================================
# [3] 데이터 로드
# ============================================================

print("\n[3] 데이터 로드")

# 체크포인트
with open(f'{backup_dir}/checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
    globals().update(checkpoint)

print("   ✅ checkpoint.pkl 로드")

# PyCaret 결과
pycaret_results = pd.read_csv(f'{backup_dir}/pycaret_results.csv')

print("   ✅ pycaret_results.csv 로드")

# 데이터 확인
print("\n로드된 데이터:")
print(f"   X_train: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
print(f"   X_test: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}")
print(f"   y_train: {len(y_train) if hasattr(y_train, '__len__') else 'N/A'}")
print(f"   y_test: {len(y_test) if hasattr(y_test, '__len__') else 'N/A'}")

print("\nPyCaret 결과:")
print(pycaret_results)

print("\n✅ 모든 데이터 로드 완료!")