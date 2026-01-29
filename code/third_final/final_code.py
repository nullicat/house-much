# ============================================================
# 🚀 2단계 모델 (분류 → 회귀) + 자동 분석 파이프라인
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor
from scipy.stats import wilcoxon
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 2단계 모델 (분류 → 회귀) + 자동 분석 파이프라인")
print("=" * 80)

# ============================================================
# [0] Checkpoint 로드
# ============================================================

print("\n[0] Checkpoint 로드")

import pickle

backup_dir = '/content/drive/MyDrive/auction_project_backup'

with open(f'{backup_dir}/checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# 데이터 추출
X_train = checkpoint['X_train']
X_test = checkpoint['X_test']
y_train = checkpoint['y_train']
y_test = checkpoint['y_test']

# Array 변환
y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

# 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")
print(f"   ✅ 데이터 준비 완료")

# Baseline 로드
baseline_pred = None
if 'final_ensemble_pred' in checkpoint:
    baseline_pred = checkpoint['final_ensemble_pred']
    print(f"   ✅ Baseline 로드 완료")
elif 'ensemble_pred' in checkpoint:
    baseline_pred = checkpoint['ensemble_pred']
    print(f"   ✅ Baseline 로드 완료 (ensemble_pred)")
else:
    print(f"   ⚠️ Baseline 없음")

# ============================================================
# [1] Stage 1: 분류 모델 (유찰 vs 낙찰)
# ============================================================

print("\n" + "=" * 80)
print("[1] Stage 1: 분류 모델 (유찰 vs 낙찰)")
print("=" * 80)

# 임계값 설정
threshold = 0.5
y_class_train = (y_train_array < threshold).astype(int)
y_class_test = (y_test_array < threshold).astype(int)

print(f"\n임계값: {threshold}")
print(f"유찰 케이스 (train): {y_class_train.sum()}개 ({y_class_train.sum()/len(y_class_train)*100:.1f}%)")
print(f"유찰 케이스 (test): {y_class_test.sum()}개 ({y_class_test.sum()/len(y_class_test)*100:.1f}%)")

# 분류 모델 (RandomForest with tuning)
print("\n분류 모델 학습 중...")

clf_params = {
    'n_estimators': 1000,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',  # 불균형 처리!
    'random_state': 42,
    'n_jobs': -1
}

clf = RandomForestClassifier(**clf_params)
clf.fit(X_train_scaled, y_class_train)

# 분류 성능 평가
y_class_pred_train = clf.predict(X_train_scaled)
y_class_pred_test = clf.predict(X_test_scaled)

print("\n[Train 분류 성능]")
print(classification_report(y_class_train, y_class_pred_train,
                          target_names=['낙찰', '유찰'],
                          digits=3))

print("\n[Test 분류 성능]")
print(classification_report(y_class_test, y_class_pred_test,
                          target_names=['낙찰', '유찰'],
                          digits=3))

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_class_pred_test)
print("\n[Confusion Matrix]")
print(f"   TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"   FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

# ============================================================
# [2] Stage 2: 그룹별 회귀 모델
# ============================================================

print("\n" + "=" * 80)
print("[2] Stage 2: 그룹별 회귀 모델")
print("=" * 80)

# Train 그룹 분리
fail_mask_train = (y_class_pred_train == 1)
success_mask_train = (y_class_pred_train == 0)

X_fail_train = X_train_scaled[fail_mask_train]
y_fail_train = y_train_array[fail_mask_train]

X_success_train = X_train_scaled[success_mask_train]
y_success_train = y_train_array[success_mask_train]

print(f"\n[Train 그룹 분리]")
print(f"   유찰 그룹: {len(X_fail_train)}개")
print(f"   낙찰 그룹: {len(X_success_train)}개")

# 2.1 유찰 그룹 모델 (저가 전용)
print("\n유찰 그룹 모델 학습 중...")

huber_fail = HuberRegressor(
    epsilon=1.1,      # 더 공격적 (이상치에 덜 민감)
    alpha=0.00001,    # 정규화 약하게
    max_iter=500
)
huber_fail.fit(X_fail_train, y_fail_train)

# 검증
fail_pred_train = huber_fail.predict(X_fail_train)
fail_mae_train = mean_absolute_error(y_fail_train, fail_pred_train)
print(f"   유찰 그룹 Train MAE: {fail_mae_train:.4f}")

# 2.2 낙찰 그룹 모델 (정상가 전용)
print("\n낙찰 그룹 모델 학습 중...")

huber_success = HuberRegressor(
    epsilon=1.35,     # 기본값
    alpha=0.0001,
    max_iter=200
)
huber_success.fit(X_success_train, y_success_train)

# 검증
success_pred_train = huber_success.predict(X_success_train)
success_mae_train = mean_absolute_error(y_success_train, success_pred_train)
print(f"   낙찰 그룹 Train MAE: {success_mae_train:.4f}")

print(f"\n✅ Stage 2 학습 완료")

# ============================================================
# [3] 2단계 예측 (Test)
# ============================================================

print("\n" + "=" * 80)
print("[3] 2단계 예측")
print("=" * 80)

# Test 그룹 분리
fail_mask_test = (y_class_pred_test == 1)
success_mask_test = (y_class_pred_test == 0)

print(f"\n[Test 그룹 분리]")
print(f"   유찰 예측: {fail_mask_test.sum()}개")
print(f"   낙찰 예측: {success_mask_test.sum()}개")

# 그룹별 예측
pred_2stage = np.zeros(len(X_test_scaled))

pred_2stage[success_mask_test] = huber_success.predict(
    X_test_scaled[success_mask_test]
)
pred_2stage[fail_mask_test] = huber_fail.predict(
    X_test_scaled[fail_mask_test]
)

print(f"\n✅ 2단계 예측 완료")

# ============================================================
# [4] 평가
# ============================================================

print("\n" + "=" * 80)
print("[4] 평가")
print("=" * 80)

# 전체 성능
mae_total = mean_absolute_error(y_test_array, pred_2stage)
print(f"\n[전체 성능]")
print(f"   2-Stage MAE: {mae_total:.4f}")

baseline_mae = 0.0715
improvement = (baseline_mae - mae_total) / baseline_mae * 100
print(f"   Baseline MAE: {baseline_mae:.4f}")
print(f"   개선: {improvement:+.1f}%")

# 저가 구간
low_mask = (y_test_array < 0.5)
mae_low = mean_absolute_error(y_test_array[low_mask], pred_2stage[low_mask])
baseline_low = 0.0805

print(f"\n[저가 구간 (실제 유찰)]")
print(f"   2-Stage MAE: {mae_low:.4f}")
print(f"   Baseline MAE: {baseline_low:.4f}")
print(f"   개선: {(baseline_low - mae_low) / baseline_low * 100:+.1f}%")

# 고가 구간
high_mask = (y_test_array >= 0.5)
mae_high = mean_absolute_error(y_test_array[high_mask], pred_2stage[high_mask])
baseline_high = 0.0710  # Baseline 고가 구간

print(f"\n[고가 구간 (실제 낙찰)]")
print(f"   2-Stage MAE: {mae_high:.4f}")
print(f"   Baseline MAE: {baseline_high:.4f}")
print(f"   개선: {(baseline_high - mae_high) / baseline_high * 100:+.1f}%")

# Within 5%p
abs_errors = np.abs(pred_2stage - y_test_array)
within_5p_total = (abs_errors <= 0.05).sum() / len(abs_errors) * 100
within_5p_low = (abs_errors[low_mask] <= 0.05).sum() / low_mask.sum() * 100
within_5p_high = (abs_errors[high_mask] <= 0.05).sum() / high_mask.sum() * 100

print(f"\n[Within 5%p]")
print(f"   전체: {within_5p_total:.1f}% (Baseline: 49.7%, {within_5p_total - 49.7:+.1f}%p)")
print(f"   저가: {within_5p_low:.1f}% (Baseline: 25.7%, {within_5p_low - 25.7:+.1f}%p)")
print(f"   고가: {within_5p_high:.1f}% (Baseline: 51.1%, {within_5p_high - 51.1:+.1f}%p)")

# Within 10%p
within_10p_total = (abs_errors <= 0.10).sum() / len(abs_errors) * 100
print(f"\n[Within 10%p]")
print(f"   전체: {within_10p_total:.1f}% (Baseline: 85.7%)")

# ============================================================
# [5] 통계적 유의성 검정
# ============================================================

print("\n" + "=" * 80)
print("[5] 통계적 유의성 검정")
print("=" * 80)

if baseline_pred is not None:
    # Wilcoxon Signed-Rank Test
    errors_baseline = np.abs(baseline_pred - y_test_array)
    errors_2stage = np.abs(pred_2stage - y_test_array)

    statistic, p_value = wilcoxon(errors_baseline, errors_2stage, alternative='greater')

    print(f"\n[Wilcoxon Signed-Rank Test]")
    print(f"   H0: Baseline ≤ 2-Stage")
    print(f"   H1: Baseline > 2-Stage (2-Stage가 더 좋음)")
    print(f"   Statistic: {statistic:.0f}")
    print(f"   p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"   ✅ 유의수준 0.05에서 유의! (2-Stage가 통계적으로 우수)")
    elif p_value < 0.10:
        print(f"   ⚠️ 유의수준 0.10에서 유의 (약한 증거)")
    else:
        print(f"   ❌ 통계적으로 유의하지 않음")

    # Cohen's d (Effect Size)
    mean_diff = errors_baseline.mean() - errors_2stage.mean()
    pooled_std = np.sqrt((errors_baseline.std()**2 + errors_2stage.std()**2) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"\n[Cohen's d (Effect Size)]")
    print(f"   Cohen's d: {cohens_d:.6f}")

    if abs(cohens_d) < 0.2:
        effect = "무시 가능 (negligible)"
    elif abs(cohens_d) < 0.5:
        effect = "작음 (small)"
    elif abs(cohens_d) < 0.8:
        effect = "중간 (medium)"
    else:
        effect = "큼 (large)"

    print(f"   효과 크기: {effect}")

    # 저가 구간 따로
    print(f"\n[저가 구간 Wilcoxon Test]")
    errors_baseline_low = errors_baseline[low_mask]
    errors_2stage_low = errors_2stage[low_mask]

    statistic_low, p_value_low = wilcoxon(errors_baseline_low, errors_2stage_low, alternative='greater')

    print(f"   Statistic: {statistic_low:.0f}")
    print(f"   p-value: {p_value_low:.6f}")

    if p_value_low < 0.05:
        print(f"   ✅ 저가 구간 개선 유의!")
    else:
        print(f"   ⚠️ 저가 구간 개선 미미")

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

# 2-Stage
categories_2stage = [categorize_error(e) for e in abs_errors]
counts_2stage = Counter(categories_2stage)

total = len(abs_errors)

print("\n[2-Stage Model]")
print(f"   Excellent (≤3%p): {counts_2stage['Excellent']/total*100:.1f}%")
print(f"   Good (3~5%p):     {counts_2stage['Good']/total*100:.1f}%")
print(f"   Fair (5~10%p):    {counts_2stage['Fair']/total*100:.1f}%")
print(f"   Poor (>10%p):     {counts_2stage['Poor']/total*100:.1f}%")

# Baseline 비교
if baseline_pred is not None:
    categories_baseline = [categorize_error(e) for e in errors_baseline]
    counts_baseline = Counter(categories_baseline)

    print("\n[Baseline (Ensemble)]")
    print(f"   Excellent (≤3%p): {counts_baseline['Excellent']/total*100:.1f}%")
    print(f"   Good (3~5%p):     {counts_baseline['Good']/total*100:.1f}%")
    print(f"   Fair (5~10%p):    {counts_baseline['Fair']/total*100:.1f}%")
    print(f"   Poor (>10%p):     {counts_baseline['Poor']/total*100:.1f}%")

    print("\n[개선율]")
    improvements = {
        'Excellent': (counts_2stage['Excellent'] - counts_baseline['Excellent'])/total*100,
        'Good': (counts_2stage['Good'] - counts_baseline['Good'])/total*100,
        'Fair': (counts_2stage['Fair'] - counts_baseline['Fair'])/total*100,
        'Poor': (counts_2stage['Poor'] - counts_baseline['Poor'])/total*100
    }

    for cat, imp in improvements.items():
        symbol = "✅" if (imp > 0 and cat in ['Excellent', 'Good']) or (imp < 0 and cat in ['Fair', 'Poor']) else "⚠️"
        print(f"   {cat:10s} {imp:+.1f}%p {symbol}")

# 저가 구간 혼동행렬
print("\n[저가 구간 혼동행렬]")
categories_2stage_low = [categorize_error(e) for e in abs_errors[low_mask]]
counts_2stage_low = Counter(categories_2stage_low)

print(f"   Excellent (≤3%p): {counts_2stage_low['Excellent']/low_mask.sum()*100:.1f}%")
print(f"   Good (3~5%p):     {counts_2stage_low['Good']/low_mask.sum()*100:.1f}%")
print(f"   Fair (5~10%p):    {counts_2stage_low['Fair']/low_mask.sum()*100:.1f}%")
print(f"   Poor (>10%p):     {counts_2stage_low['Poor']/low_mask.sum()*100:.1f}%")

# ============================================================
# [7] 시각화 (12개)
# ============================================================

print("\n[7] 시각화")

fig = plt.figure(figsize=(24, 16))

# 1. 분류 Confusion Matrix
ax1 = plt.subplot(4, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['낙찰', '유찰'],
            yticklabels=['낙찰', '유찰'],
            ax=ax1, cbar=False)
ax1.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=10, fontweight='bold')
ax1.set_title('Stage 1: 분류 성능', fontsize=11, fontweight='bold')

# 2. 혼동행렬 비교 (Baseline)
if baseline_pred is not None:
    ax2 = plt.subplot(4, 3, 2)
    categories = ['Excellent', 'Good', 'Fair', 'Poor']
    baseline_pcts = [counts_baseline[c]/total*100 for c in categories]

    bars = ax2.bar(range(len(categories)), baseline_pcts,
                   color=['green', 'lightgreen', 'orange', 'red'],
                   edgecolor='black', linewidth=2, alpha=0.8)

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Baseline\nWithin 5%p: 49.7%', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim(0, 50)

    for bar, pct in zip(bars, baseline_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, pct + 1,
                 f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

# 3. 혼동행렬 (2-Stage)
ax3 = plt.subplot(4, 3, 3)
stage_pcts = [counts_2stage[c]/total*100 for c in categories]

bars = ax3.bar(range(len(categories)), stage_pcts,
               color=['green', 'lightgreen', 'orange', 'red'],
               edgecolor='black', linewidth=2, alpha=0.8)

ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax3.set_title(f'2-Stage Model\nWithin 5%p: {within_5p_total:.1f}%',
              fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim(0, 50)

for bar, pct in zip(bars, stage_pcts):
    ax3.text(bar.get_x() + bar.get_width()/2, pct + 1,
             f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

# 4. 오차 분포 비교
if baseline_pred is not None:
    ax4 = plt.subplot(4, 3, 4)
    ax4.hist(errors_baseline, bins=50, alpha=0.5, label='Baseline',
             color='blue', edgecolor='black')
    ax4.hist(abs_errors, bins=50, alpha=0.5, label='2-Stage',
             color='red', edgecolor='black')
    ax4.axvline(0.05, color='green', linestyle='--', linewidth=2, label='5%p')
    ax4.set_xlabel('Absolute Error', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('오차 분포 비교', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

# 5. 예측 vs 실제 (전체)
ax5 = plt.subplot(4, 3, 5)
ax5.scatter(y_test_array, pred_2stage, alpha=0.3, s=10)
ax5.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect')
ax5.set_xlabel('실제값', fontsize=10, fontweight='bold')
ax5.set_ylabel('예측값', fontsize=10, fontweight='bold')
ax5.set_title(f'예측 vs 실제 (전체)\nMAE: {mae_total:.4f}',
              fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. 예측 vs 실제 (저가)
ax6 = plt.subplot(4, 3, 6)
ax6.scatter(y_test_array[low_mask], pred_2stage[low_mask],
            alpha=0.5, s=20, color='red')
ax6.plot([0, 0.5], [0, 0.5], 'r--', linewidth=2, label='Perfect')
ax6.set_xlabel('실제값', fontsize=10, fontweight='bold')
ax6.set_ylabel('예측값', fontsize=10, fontweight='bold')
ax6.set_title(f'예측 vs 실제 (저가)\nMAE: {mae_low:.4f}',
              fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. 누적 분포 (CDF)
ax7 = plt.subplot(4, 3, 7)
sorted_errors = np.sort(abs_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

ax7.plot(sorted_errors, cdf, linewidth=2, label='2-Stage')
if baseline_pred is not None:
    sorted_baseline = np.sort(errors_baseline)
    cdf_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline) * 100
    ax7.plot(sorted_baseline, cdf_baseline, linewidth=2, alpha=0.7, label='Baseline')

ax7.axvline(0.05, color='green', linestyle='--', linewidth=2)
ax7.axvline(0.10, color='orange', linestyle='--', linewidth=2)
ax7.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax7.set_xlabel('Absolute Error', fontsize=10, fontweight='bold')
ax7.set_ylabel('Cumulative %', fontsize=10, fontweight='bold')
ax7.set_title('누적 오차 분포 (CDF)', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. 구간별 성능
ax8 = plt.subplot(4, 3, 8)
bins = [0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 1.0]
labels = ['0-1%', '1-3%', '3-5%', '5-7%', '7-10%', '10-15%', '15-20%', '>20%']
error_dist = pd.cut(abs_errors, bins=bins, labels=labels).value_counts().sort_index()

bars = ax8.bar(range(len(error_dist)), error_dist.values / len(abs_errors) * 100,
               color=['darkgreen', 'green', 'lightgreen', 'yellow',
                      'orange', 'darkorange', 'red', 'darkred'],
               edgecolor='black', alpha=0.8)

ax8.set_xticks(range(len(error_dist)))
ax8.set_xticklabels(labels, rotation=45, fontsize=8)
ax8.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax8.set_title('구간별 오차 분포', fontsize=11, fontweight='bold')
ax8.grid(alpha=0.3, axis='y')

for bar, val in zip(bars, error_dist.values / len(abs_errors) * 100):
    if val > 0.5:
        ax8.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.1f}%', ha='center', fontsize=7)

# 9. 저가 구간 혼동행렬
ax9 = plt.subplot(4, 3, 9)
low_pcts = [counts_2stage_low[c]/low_mask.sum()*100 for c in categories]

bars = ax9.bar(range(len(categories)), low_pcts,
               color=['green', 'lightgreen', 'orange', 'red'],
               edgecolor='black', linewidth=2, alpha=0.8)

ax9.set_xticks(range(len(categories)))
ax9.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax9.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax9.set_title(f'저가 구간 혼동행렬\nWithin 5%p: {within_5p_low:.1f}%',
              fontsize=11, fontweight='bold')
ax9.grid(alpha=0.3, axis='y')
ax9.set_ylim(0, 50)

for bar, pct in zip(bars, low_pcts):
    ax9.text(bar.get_x() + bar.get_width()/2, pct + 1,
             f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

# 10. 그룹별 성능 비교
ax10 = plt.subplot(4, 3, 10)
groups = ['전체', '저가', '고가']
baseline_vals = [0.0715, 0.0805, 0.0710]
stage_vals = [mae_total, mae_low, mae_high]

x = np.arange(len(groups))
width = 0.35

bars1 = ax10.bar(x - width/2, baseline_vals, width, label='Baseline',
                 color='skyblue', edgecolor='black', alpha=0.8)
bars2 = ax10.bar(x + width/2, stage_vals, width, label='2-Stage',
                 color='salmon', edgecolor='black', alpha=0.8)

ax10.set_ylabel('MAE', fontsize=10, fontweight='bold')
ax10.set_title('그룹별 MAE 비교', fontsize=11, fontweight='bold')
ax10.set_xticks(x)
ax10.set_xticklabels(groups, fontsize=9, fontweight='bold')
ax10.legend()
ax10.grid(alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# 11. Within 5%p 비교
ax11 = plt.subplot(4, 3, 11)
groups = ['전체', '저가', '고가']
baseline_vals = [49.7, 25.7, 51.1]
stage_vals = [within_5p_total, within_5p_low, within_5p_high]

x = np.arange(len(groups))

bars1 = ax11.bar(x - width/2, baseline_vals, width, label='Baseline',
                 color='skyblue', edgecolor='black', alpha=0.8)
bars2 = ax11.bar(x + width/2, stage_vals, width, label='2-Stage',
                 color='salmon', edgecolor='black', alpha=0.8)

ax11.set_ylabel('Within 5%p (%)', fontsize=10, fontweight='bold')
ax11.set_title('그룹별 Within 5%p 비교', fontsize=11, fontweight='bold')
ax11.set_xticks(x)
ax11.set_xticklabels(groups, fontsize=9, fontweight='bold')
ax11.legend()
ax11.grid(alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# 12. 개선율 요약
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')

if baseline_pred is not None:
    summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        2단계 모델 개선 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[전체 성능]
MAE: {baseline_mae:.4f} → {mae_total:.4f}
개선: {improvement:+.1f}%

[저가 구간]
MAE: {baseline_low:.4f} → {mae_low:.4f}
개선: {(baseline_low - mae_low) / baseline_low * 100:+.1f}%

Within 5%p: {25.7:.1f}% → {within_5p_low:.1f}%
개선: {within_5p_low - 25.7:+.1f}%p

[Within 5%p]
전체: {49.7:.1f}% → {within_5p_total:.1f}%
개선: {within_5p_total - 49.7:+.1f}%p

[혼동행렬]
Fair: {counts_baseline['Fair']/total*100:.1f}% → {counts_2stage['Fair']/total*100:.1f}%
Poor: {counts_baseline['Poor']/total*100:.1f}% → {counts_2stage['Poor']/total*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
else:
    summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        2단계 모델 성능 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[전체 성능]
MAE: {mae_total:.4f}
개선: {improvement:+.1f}%

[저가 구간] ⭐
MAE: {mae_low:.4f}
개선: {(baseline_low - mae_low) / baseline_low * 100:+.1f}%

Within 5%p: {within_5p_low:.1f}%
개선: {within_5p_low - 25.7:+.1f}%p

[Within 5%p]
전체: {within_5p_total:.1f}%
개선: {within_5p_total - 49.7:+.1f}%p

[저가 혼동행렬]
Excellent: {counts_2stage_low['Excellent']/low_mask.sum()*100:.1f}%
Good: {counts_2stage_low['Good']/low_mask.sum()*100:.1f}%
Fair: {counts_2stage_low['Fair']/low_mask.sum()*100:.1f}%
Poor: {counts_2stage_low['Poor']/low_mask.sum()*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
          verticalalignment='center')

plt.tight_layout()
plt.savefig(f'{backup_dir}/2stage_results_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   ✅ 저장: 2stage_results_complete.png")

# ============================================================
# [8] 최종 요약
# ============================================================

print("\n" + "=" * 80)
print("🏆 최종 요약")
print("=" * 80)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                2단계 모델 최종 성능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[전체 성능]
Baseline MAE:       {baseline_mae:.4f}
2-Stage MAE:        {mae_total:.4f}
개선:               {improvement:+.1f}%

[저가 구간] ⭐
Baseline MAE:       {baseline_low:.4f}
2-Stage MAE:        {mae_low:.4f}
개선:               {(baseline_low - mae_low) / baseline_low * 100:+.1f}%

[고가 구간]
Baseline MAE:       {baseline_high:.4f}
2-Stage MAE:        {mae_high:.4f}
개선:               {(baseline_high - mae_high) / baseline_high * 100:+.1f}%

[Within 5%p]
Baseline (전체):    49.7%
2-Stage (전체):     {within_5p_total:.1f}% ({within_5p_total - 49.7:+.1f}%p)

Baseline (저가):    25.7%
2-Stage (저가):     {within_5p_low:.1f}% ({within_5p_low - 25.7:+.1f}%p) ⭐

Baseline (고가):    51.1%
2-Stage (고가):     {within_5p_high:.1f}% ({within_5p_high - 51.1:+.1f}%p)

[Within 10%p]
2-Stage:            {within_10p_total:.1f}%
Baseline:           85.7%

[혼동행렬]
Excellent + Good:   {(counts_2stage['Excellent'] + counts_2stage['Good'])/total*100:.1f}%
Fair:               {counts_2stage['Fair']/total*100:.1f}%
Poor:               {counts_2stage['Poor']/total*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 평가
if improvement > 3:
    print("🎉🎉🎉 전체 성능 개선! 2단계 모델 효과 확인!")
elif improvement > 1:
    print("✅ 전체 성능 소폭 개선")
else:
    print("⚠️ 전체 성능 비슷")

if (baseline_low - mae_low) / baseline_low * 100 > 10:
    print("🎉🎉🎉 저가 구간 대폭 개선! 2단계 모델의 핵심 성과!")
elif (baseline_low - mae_low) / baseline_low * 100 > 5:
    print("✅✅ 저가 구간 개선!")
else:
    print("⚠️ 저가 구간 개선 미미")

if within_5p_low - 25.7 > 10:
    print(f"🎉🎉🎉 저가 Within 5%p 대폭 개선! (+{within_5p_low - 25.7:.1f}%p)")
elif within_5p_low - 25.7 > 5:
    print(f"✅✅ 저가 Within 5%p 개선! (+{within_5p_low - 25.7:.1f}%p)")

print("\n" + "=" * 80)

# ============================================================
# [9] 결과 저장
# ============================================================

print("\n[9] 결과 저장")

# 모델 저장
import joblib

joblib.dump(clf, f'{backup_dir}/2stage_classifier.pkl')
joblib.dump(huber_fail, f'{backup_dir}/2stage_huber_fail.pkl')
joblib.dump(huber_success, f'{backup_dir}/2stage_huber_success.pkl')

print(f"   ✅ 모델 저장 완료")

# 예측값 저장
stage2_predictions = {
    'y_test': y_test_array,
    'pred': pred_2stage,
    'classifier_pred': y_class_pred_test,
    'mae_total': mae_total,
    'mae_low': mae_low,
    'mae_high': mae_high,
    'within_5p_total': within_5p_total,
    'within_5p_low': within_5p_low,
    'within_5p_high': within_5p_high,
    'within_10p_total': within_10p_total,
    'counts_2stage': counts_2stage,
    'counts_2stage_low': counts_2stage_low,
    'improvements': improvements if baseline_pred is not None else None
}

# Checkpoint에 추가
checkpoint['stage2_predictions'] = stage2_predictions

# 저장
with open(f'{backup_dir}/checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

print(f"   ✅ 결과 저장 완료: checkpoint.pkl에 추가")
print("\n" + "=" * 80)