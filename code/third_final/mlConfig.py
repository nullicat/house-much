# ============================================================
# 🔍 ML 모델 재검증 (정규화 포함)
# ============================================================

print("=" * 80)
print("🔍 ML 모델 재검증 (정규화)")
print("=" * 80)

print("""
의심:
checkpoint의 다른 ML 모델들도
정규화 안 해서 성능 낮았을 가능성!

재검증:
- Linear, Ridge, Lasso
- Random Forest
- XGBoost, LightGBM
- CatBoost

모두 정규화해서 재학습!
""")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# [1] 정규화된 데이터 (이미 있음)
# ============================================================

print("\n[1] 데이터 준비")
print(f"   정규화된 Train: {X_train_scaled.shape}")
print(f"   정규화된 Test: {X_test_scaled.shape}")

# ============================================================
# [2] 모델 재학습
# ============================================================

print("\n[2] 모델 재학습 (정규화 데이터)")
print("   (약 2~3분 소요)")

models_to_test = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'Huber': HuberRegressor(epsilon=1.35, alpha=0.0001),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=8, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=8, random_state=42, verbose=-1),
}

ml_results = []

print("\n결과:")
print("-" * 70)

for name, model in models_to_test.items():
    # 학습
    model.fit(X_train_scaled, y_train_array)

    # 예측
    pred = model.predict(X_test_scaled)

    # 평가
    mae = mean_absolute_error(y_test_array, pred)
    rmse = np.sqrt(mean_squared_error(y_test_array, pred))
    r2 = r2_score(y_test_array, pred)

    ml_results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Data': '24개 (정규화)'
    })

    print(f"   {name:20s} MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

# CatBoost (10개 피처, 정규화 안 함)
ml_results.append({
    'Model': 'CatBoost',
    'MAE': catboost_test_mae,
    'RMSE': catboost_test_rmse,
    'R²': catboost_test_r2,
    'Data': '10개 (선택)'
})

# Ensemble
ml_results.append({
    'Model': 'Ensemble (Huber+CatBoost)',
    'MAE': best_ensemble_mae,
    'RMSE': np.sqrt(mean_squared_error(y_test_array, final_ensemble_pred)),
    'R²': final_ensemble_r2,
    'Data': '24개 + 10개'
})

# ============================================================
# [3] 비교 분석
# ============================================================

print("\n[3] 전체 비교")

ml_results_df = pd.DataFrame(ml_results).sort_values('MAE')

print("\n📊 ML 모델 순위 (정규화 포함):")
print(ml_results_df.to_string(index=False))

# 최고 모델
best_ml = ml_results_df.iloc[0]

print(f"\n🏆 최고 ML 모델:")
print(f"   모델: {best_ml['Model']}")
print(f"   MAE: {best_ml['MAE']:.4f}")
print(f"   R²: {best_ml['R²']:.4f}")

# ============================================================
# [4] 충격 분석
# ============================================================

print("\n[4] 충격적 발견?")

# Linear 계열 확인
linear_models = ml_results_df[ml_results_df['Model'].isin(['Linear Regression', 'Ridge', 'Lasso', 'Huber'])]

print(f"\nLinear 계열 성능:")
print(linear_models[['Model', 'MAE', 'R²']].to_string(index=False))

best_linear = linear_models.iloc[0]

if best_linear['MAE'] < 0.0753:
    print(f"\n   💥 충격! {best_linear['Model']}이 CatBoost(0.0753)보다 좋음!")
    print(f"   → 당시 정규화 안 해서 놓쳤을 가능성!")
else:
    print(f"\n   → CatBoost가 여전히 우수")

# ============================================================
# [5] 시각화
# ============================================================

print("\n[5] 시각화")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. MAE 비교
ax1 = axes[0]

models = ml_results_df['Model']
maes = ml_results_df['MAE']

bars = ax1.barh(range(len(models)), maes)

# 색상 (낮을수록 녹색)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(models)))
for bar, color in zip(bars, colors):
    bar.set_color(color)
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)

ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models, fontsize=10)
ax1.set_xlabel('MAE', fontsize=12, fontweight='bold')
ax1.set_title('ML 모델 비교 (정규화 포함)', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# 값 표시
for i, mae in enumerate(maes):
    ax1.text(mae + 0.002, i, f"{mae:.4f}", va='center', fontsize=9)

# Baseline
ax1.axvline(x=0.1402, color='red', linestyle='--', linewidth=2,
            label='Baseline', alpha=0.7)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# 2. R² 비교
ax2 = axes[1]

r2_values = ml_results_df['R²']
bars = ax2.barh(range(len(models)), r2_values)

for bar, color in zip(bars, colors):
    bar.set_color(color)
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)

ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=10)
ax2.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax2.set_title('R² 비교', fontsize=13, fontweight='bold')
ax2.invert_yaxis()

for i, r2 in enumerate(r2_values):
    ax2.text(r2 + 0.01, i, f"{r2:.4f}", va='center', fontsize=9)

ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{backup_dir}/ml_models_revalidation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ 저장: ml_models_revalidation.png")

# ============================================================
# 결론
# ============================================================

print("\n" + "=" * 80)
print("🎯 결론")
print("=" * 80)

if best_ml['Model'] in ['Linear Regression', 'Ridge', 'Lasso', 'Huber']:
    if best_ml['MAE'] < 0.0753:
        print(f"""
💥 충격적 발견!

{best_ml['Model']}이 CatBoost보다 우수!
- {best_ml['Model']}: {best_ml['MAE']:.4f}
- CatBoost: 0.0753

원인:
→ 당시 Linear 모델들을 정규화 안 함
→ 성능이 낮게 측정됨
→ CatBoost만 선택

교훈:
데이터 전처리가 모델 선택에 결정적!
        """)
    else:
        print(f"""
예상대로!

Linear 모델들도 정규화하면 좋아지지만,
여전히 CatBoost/Ensemble이 최고!

- {best_ml['Model']}: {best_ml['MAE']:.4f}
- CatBoost: {catboost_test_mae:.4f}
- Ensemble: {best_ensemble_mae:.4f}

→ 초기 분석이 옳았음!
        """)
else:
    print(f"""
예상 외!

{best_ml['Model']}이 최고!

재평가 필요할 수도...
    """)

print("✅ ML 모델 재검증 완료!")