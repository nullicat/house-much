# ============================================================
# 🚀 TabNet - Tabular 데이터 전용 딥러닝
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 TabNet - Tabular 전용 딥러닝")
print("=" * 80)

# ============================================================
# [0] 설치 확인
# ============================================================

print("\n[0] TabNet 설치 확인")

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ TabNet: Installed")
except ImportError:
    print("   ⚠️ TabNet 설치 필요")
    print("   설치 중...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pytorch-tabnet', '-q'])
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    print(f"   ✅ 설치 완료!")

# CUDA 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

# ============================================================
# [1] Checkpoint 로드
# ============================================================

print("\n[1] Checkpoint 로드")

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
X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# 스케일링 (TabNet은 스케일링 권장)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_array)
X_test_scaled = scaler.transform(X_test_array)

print(f"   ✅ 데이터 준비 완료")

# ============================================================
# [2] TabNet 모델 구성
# ============================================================

print("\n[2] TabNet 모델 구성")

# 하이퍼파라미터
tabnet_params = {
    'n_d': 64,  # Feature dimension for decision
    'n_a': 64,  # Feature dimension for attention
    'n_steps': 5,  # Number of decision steps
    'gamma': 1.5,  # Coefficient for feature reusage
    'n_independent': 2,  # Number of independent GLU layers
    'n_shared': 2,  # Number of shared GLU layers
    'lambda_sparse': 0.0001,  # Sparsity regularization
    'momentum': 0.3,  # Momentum for batch normalization
    'clip_value': 2.0,  # Gradient clipping
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=0.02),
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'scheduler_params': {"step_size": 50, "gamma": 0.9},
    'mask_type': 'entmax',  # 'sparsemax' or 'entmax'
    'seed': 42,
    'verbose': 10,
    'device_name': device
}

print(f"   n_d (Feature dim): {tabnet_params['n_d']}")
print(f"   n_a (Attention dim): {tabnet_params['n_a']}")
print(f"   n_steps (Decision steps): {tabnet_params['n_steps']}")
print(f"   gamma (Feature reuse): {tabnet_params['gamma']}")

# 모델 생성
tabnet = TabNetRegressor(**tabnet_params)

print(f"   ✅ TabNet 모델 생성 완료")

# ============================================================
# [3] 학습
# ============================================================

print("\n[3] 학습 시작 (약 5~10분 소요)")

# Train/Validation 분리
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_array,
    test_size=0.2,
    random_state=42
)

print(f"   Train: {X_train_split.shape}")
print(f"   Val: {X_val_split.shape}")

# 학습
tabnet.fit(
    X_train=X_train_split,
    y_train=y_train_split.reshape(-1, 1),
    eval_set=[(X_val_split, y_val_split.reshape(-1, 1))],
    eval_name=['val'],
    eval_metric=['mae'],
    max_epochs=200,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    loss_fn=torch.nn.HuberLoss(delta=1.35)  # Huber Loss 사용!
)

print(f"\n   ✅ 학습 완료")

# ============================================================
# [4] 평가
# ============================================================

print("\n" + "=" * 80)
print("[4] 평가")
print("=" * 80)

# 예측
pred_tabnet = tabnet.predict(X_test_scaled).flatten()

# 전체 성능
mae_total = mean_absolute_error(y_test_array, pred_tabnet)
print(f"\n   전체 MAE: {mae_total:.4f}")

baseline_mae = 0.0715
improvement = (baseline_mae - mae_total) / baseline_mae * 100
print(f"   Baseline: {baseline_mae:.4f}")
print(f"   개선: {improvement:+.1f}%")

# 저가 구간
low_mask = (y_test_array < 0.5)
mae_low = mean_absolute_error(y_test_array[low_mask], pred_tabnet[low_mask])
baseline_low = 0.0805

print(f"\n   저가 MAE: {mae_low:.4f}")
print(f"   Baseline: {baseline_low:.4f}")
print(f"   개선: {(baseline_low - mae_low) / baseline_low * 100:+.1f}%")

# Within 5%p
abs_errors = np.abs(pred_tabnet - y_test_array)
within_5p_total = (abs_errors <= 0.05).sum() / len(abs_errors) * 100
within_5p_low = (abs_errors[low_mask] <= 0.05).sum() / low_mask.sum() * 100

print(f"\n   전체 Within 5%p: {within_5p_total:.1f}%")
print(f"   Baseline: 49.7%")
print(f"   개선: {within_5p_total - 49.7:+.1f}%p")

print(f"\n   저가 Within 5%p: {within_5p_low:.1f}%")
print(f"   Baseline: 25.7%")
print(f"   개선: {within_5p_low - 25.7:+.1f}%p")

# Within 10%p
within_10p_total = (abs_errors <= 0.10).sum() / len(abs_errors) * 100
print(f"\n   전체 Within 10%p: {within_10p_total:.1f}%")
print(f"   Baseline: 85.7%")

# ============================================================
# [5] Feature Importance (TabNet의 강점!)
# ============================================================

print("\n" + "=" * 80)
print("[5] Feature Importance")
print("=" * 80)

# Feature Importance 추출
importance = tabnet.feature_importances_

# 정렬
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n[Top 10 Features]")
print(feature_importance.head(10).to_string(index=False))

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

tabnet_categories = [categorize_error(e) for e in abs_errors]
tabnet_counts = Counter(tabnet_categories)

total = len(abs_errors)

print("\n[TabNet]")
print(f"   Excellent (≤3%p): {tabnet_counts['Excellent']/total*100:.1f}%")
print(f"   Good (3~5%p):     {tabnet_counts['Good']/total*100:.1f}%")
print(f"   Fair (5~10%p):    {tabnet_counts['Fair']/total*100:.1f}%")
print(f"   Poor (>10%p):     {tabnet_counts['Poor']/total*100:.1f}%")

# Baseline 비교
if 'final_ensemble_pred' in checkpoint or 'ensemble_pred' in checkpoint:
    baseline_pred = checkpoint.get('final_ensemble_pred') or checkpoint.get('ensemble_pred')
    if baseline_pred is not None:
        errors_baseline = np.abs(baseline_pred - y_test_array)
        baseline_categories = [categorize_error(e) for e in errors_baseline]
        baseline_counts = Counter(baseline_categories)

        print("\n[Baseline (Ensemble)]")
        print(f"   Excellent (≤3%p): {baseline_counts['Excellent']/total*100:.1f}%")
        print(f"   Good (3~5%p):     {baseline_counts['Good']/total*100:.1f}%")
        print(f"   Fair (5~10%p):    {baseline_counts['Fair']/total*100:.1f}%")
        print(f"   Poor (>10%p):     {baseline_counts['Poor']/total*100:.1f}%")

        print("\n[개선율]")
        improvements = {
            'Excellent': (tabnet_counts['Excellent'] - baseline_counts['Excellent'])/total*100,
            'Good': (tabnet_counts['Good'] - baseline_counts['Good'])/total*100,
            'Fair': (tabnet_counts['Fair'] - baseline_counts['Fair'])/total*100,
            'Poor': (tabnet_counts['Poor'] - baseline_counts['Poor'])/total*100
        }

        for cat, imp in improvements.items():
            symbol = "✅" if (imp > 0 and cat in ['Excellent', 'Good']) or (imp < 0 and cat in ['Fair', 'Poor']) else "⚠️"
            print(f"   {cat:10s} {imp:+.1f}%p {symbol}")

# ============================================================
# [7] 시각화
# ============================================================

print("\n[7] 시각화")

fig = plt.figure(figsize=(20, 12))

# 1. 학습 곡선
ax1 = plt.subplot(3, 3, 1)

# TabNet의 학습 이력 추출
history_df = pd.DataFrame(tabnet.history)

if 'loss' in history_df.columns:
    ax1.plot(history_df['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history_df['val_0_mae'], label='Val MAE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss / MAE', fontsize=10, fontweight='bold')
    ax1.set_title('학습 곡선', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

# 2. Feature Importance
ax2 = plt.subplot(3, 3, 2)
top_features = feature_importance.head(15)
ax2.barh(range(len(top_features)), top_features['Importance'].values,
         color='steelblue', edgecolor='black', alpha=0.8)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['Feature'].values, fontsize=8)
ax2.set_xlabel('Importance', fontsize=10, fontweight='bold')
ax2.set_title('Feature Importance (Top 15)', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3, axis='x')

# 3. 혼동행렬 (TabNet)
ax3 = plt.subplot(3, 3, 4)
categories = ['Excellent', 'Good', 'Fair', 'Poor']
tabnet_pcts = [tabnet_counts[c]/total*100 for c in categories]

bars = ax3.bar(range(len(categories)), tabnet_pcts,
               color=['green', 'lightgreen', 'orange', 'red'],
               edgecolor='black', linewidth=2, alpha=0.8)

ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax3.set_title(f'TabNet\nWithin 5%p: {within_5p_total:.1f}%',
              fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim(0, 50)

for bar, pct in zip(bars, tabnet_pcts):
    ax3.text(bar.get_x() + bar.get_width()/2, pct + 1,
             f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

# 4. 오차 분포
ax4 = plt.subplot(3, 3, 5)
ax4.hist(abs_errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax4.axvline(0.05, color='green', linestyle='--', linewidth=2, label='5%p')
ax4.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='10%p')
ax4.set_xlabel('Absolute Error', fontsize=10, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax4.set_title('오차 분포', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. 예측 vs 실제 (전체)
ax5 = plt.subplot(3, 3, 6)
ax5.scatter(y_test_array, pred_tabnet, alpha=0.3, s=10)
ax5.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect')
ax5.set_xlabel('실제값', fontsize=10, fontweight='bold')
ax5.set_ylabel('예측값', fontsize=10, fontweight='bold')
ax5.set_title(f'예측 vs 실제 (전체)\nMAE: {mae_total:.4f}',
              fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. 예측 vs 실제 (저가)
ax6 = plt.subplot(3, 3, 7)
ax6.scatter(y_test_array[low_mask], pred_tabnet[low_mask],
            alpha=0.5, s=20, color='red')
ax6.plot([0, 0.5], [0, 0.5], 'r--', linewidth=2, label='Perfect')
ax6.set_xlabel('실제값', fontsize=10, fontweight='bold')
ax6.set_ylabel('예측값', fontsize=10, fontweight='bold')
ax6.set_title(f'예측 vs 실제 (저가)\nMAE: {mae_low:.4f}',
              fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. 오차 누적 분포 (CDF)
ax7 = plt.subplot(3, 3, 8)
sorted_errors = np.sort(abs_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

ax7.plot(sorted_errors, cdf, linewidth=2)
ax7.axvline(0.05, color='green', linestyle='--', linewidth=2, label='5%p')
ax7.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='10%p')
ax7.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax7.set_xlabel('Absolute Error', fontsize=10, fontweight='bold')
ax7.set_ylabel('Cumulative %', fontsize=10, fontweight='bold')
ax7.set_title('누적 오차 분포 (CDF)', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. 구간별 성능
ax8 = plt.subplot(3, 3, 9)
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
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f'{val:.1f}%', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(f'{backup_dir}/tabnet_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"   ✅ 저장: tabnet_results.png")

# ============================================================
# [8] 최종 요약
# ============================================================

print("\n" + "=" * 80)
print("🏆 최종 요약")
print("=" * 80)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                TabNet 성능 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[전체 성능]
Baseline MAE:       {baseline_mae:.4f}
TabNet MAE:         {mae_total:.4f}
개선:               {improvement:+.1f}%

[저가 성능]
Baseline MAE:       {baseline_low:.4f}
TabNet MAE:         {mae_low:.4f}
개선:               {(baseline_low - mae_low) / baseline_low * 100:+.1f}%

[Within 5%p]
Baseline (전체):    49.7%
TabNet (전체):      {within_5p_total:.1f}% ({within_5p_total - 49.7:+.1f}%p)

Baseline (저가):    25.7%
TabNet (저가):      {within_5p_low:.1f}% ({within_5p_low - 25.7:+.1f}%p)

[혼동행렬]
Excellent + Good:   {(tabnet_counts['Excellent'] + tabnet_counts['Good'])/total*100:.1f}%
Fair:               {tabnet_counts['Fair']/total*100:.1f}%
Poor:               {tabnet_counts['Poor']/total*100:.1f}%

[Within 10%p]
TabNet:             {within_10p_total:.1f}%
Baseline:           85.7%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 평가
if improvement > 5:
    print("🎉🎉🎉 대폭 개선! TabNet 효과 확인!")
elif improvement > 2:
    print("✅✅ 개선됨! TabNet 효과 있음!")
elif improvement > 0:
    print("✅ 소폭 개선")
else:
    print("⚠️ 개선 안 됨. 하이퍼파라미터 튜닝 필요")

if within_5p_low > 35:
    print(f"🎉🎉🎉 저가 구간 대폭 개선! (+{within_5p_low - 25.7:.1f}%p)")
elif within_5p_low > 30:
    print(f"✅✅ 저가 구간 개선! (+{within_5p_low - 25.7:.1f}%p)")

if tabnet_counts['Poor']/total*100 < 12:
    print("🎉🎉🎉 Poor 구간 크게 감소! 목표 달성!")
elif tabnet_counts['Poor']/total*100 < 14:
    print("✅✅ Poor 구간 감소!")

print("\n" + "=" * 80)

# ============================================================
# [9] 결과 저장
# ============================================================

print("\n[9] 결과 저장")

# TabNet 모델 저장
tabnet.save_model(f'{backup_dir}/tabnet_model')
print(f"   ✅ TabNet 모델 저장: tabnet_model.zip")

# 예측값 저장
tabnet_predictions = {
    'y_test': y_test_array,
    'pred': pred_tabnet,
    'mae_total': mae_total,
    'mae_low': mae_low,
    'within_5p_total': within_5p_total,
    'within_5p_low': within_5p_low,
    'within_10p_total': within_10p_total,
    'tabnet_counts': tabnet_counts,
    'feature_importance': feature_importance
}

# Checkpoint에 추가
checkpoint['tabnet_predictions'] = tabnet_predictions

# 저장
with open(f'{backup_dir}/checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

print(f"   ✅ 결과 저장 완료: checkpoint.pkl에 추가")
print("\n" + "=" * 80)