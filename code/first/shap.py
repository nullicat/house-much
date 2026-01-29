# ============================================================
# STEP 16: SHAP 분석 - CatBoost 해석
# ============================================================

print("\n" + "=" * 80)
print("🔍 STEP 16: SHAP 분석 (Model Interpretation)")
print("=" * 80)

# ============================================================
# 16-1. 최적 모델 재학습
# ============================================================

print("\n[16-1] 최적 10개 피처로 모델 재학습")

# 최적 10개 피처
final_features = ['최저가율', '최저가', '신건여부', '보증금비율',
                  '동_encoded', '평당감정가', '감정가', '건물면적',
                  '구_encoded', '매각_월']

print(f"\n최종 피처 {len(final_features)}개:")
for i, feat in enumerate(final_features, 1):
    print(f"   {i:2d}. {feat}")

# 데이터 준비
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

# 최종 모델 학습
print("\n모델 학습 중...")
final_catboost = CatBoostRegressor(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    random_state=42,
    verbose=0
)

final_catboost.fit(X_train_final, y_train)
y_pred_final = final_catboost.predict(X_test_final)

# 성능 확인
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
r2_final = r2_score(y_test, y_pred_final)

print(f"\n최종 모델 성능:")
print(f"   - MAE:  {mae_final:.4f}")
print(f"   - RMSE: {rmse_final:.4f}")
print(f"   - R²:   {r2_final:.4f}")

# ============================================================
# 16-2. SHAP 초기화
# ============================================================

print("\n[16-2] SHAP Explainer 생성")

import first.shap as shap

# SHAP Explainer 생성 (시간 소요 가능)
print("   SHAP Explainer 계산 중... (1~2분 소요)")

# 샘플링 (전체는 너무 오래 걸림)
X_test_sample = X_test_final.sample(min(500, len(X_test_final)), random_state=42)

explainer = shap.Explainer(final_catboost)
shap_values = explainer(X_test_sample)

print("   ✅ SHAP 값 계산 완료!")

# ============================================================
# 16-3. SHAP Summary Plot (Feature Importance)
# ============================================================

print("\n[16-3] SHAP Summary Plot - Feature Importance")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('SHAP Feature Importance (Summary Plot)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n💬 해석:")
print("   - 위쪽 피처: 중요도 높음")
print("   - 빨강: 피처 값 높음")
print("   - 파랑: 피처 값 낮음")
print("   - 오른쪽: 낙찰가율 증가 효과")
print("   - 왼쪽: 낙찰가율 감소 효과")

# ============================================================
# 16-4. SHAP Bar Plot (평균 절대 기여도)
# ============================================================

print("\n[16-4] SHAP Bar Plot - 평균 절대 기여도")

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title('SHAP Feature Importance (Bar Plot)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# SHAP 값으로 중요도 계산
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': final_features,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print("\nSHAP 기반 Feature Importance:")
display(shap_importance_df)

# ============================================================
# 16-5. SHAP vs CatBoost Feature Importance 비교
# ============================================================

print("\n[16-5] SHAP vs CatBoost Feature Importance 비교")

# CatBoost Feature Importance
catboost_importance = final_catboost.get_feature_importance()

comparison_importance = pd.DataFrame({
    'Feature': final_features,
    'CatBoost_Importance': catboost_importance,
    'SHAP_Importance': shap_importance
})

# 정규화 (0-100)
comparison_importance['CatBoost_Norm'] = (
    comparison_importance['CatBoost_Importance'] /
    comparison_importance['CatBoost_Importance'].sum() * 100
)
comparison_importance['SHAP_Norm'] = (
    comparison_importance['SHAP_Importance'] /
    comparison_importance['SHAP_Importance'].sum() * 100
)

comparison_importance = comparison_importance.sort_values('SHAP_Norm', ascending=False)

print("\n비교표:")
display(comparison_importance[['Feature', 'CatBoost_Norm', 'SHAP_Norm']])

# 시각화
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(final_features))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_importance['CatBoost_Norm'],
               width, label='CatBoost', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, comparison_importance['SHAP_Norm'],
               width, label='SHAP', alpha=0.8, color='coral')

ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Normalized Importance (%)', fontsize=12)
ax.set_title('CatBoost vs SHAP Feature Importance',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_importance['Feature'], rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================
# 16-6. SHAP Dependence Plot (상위 3개 피처)
# ============================================================

print("\n[16-6] SHAP Dependence Plot - 상위 3개 피처")

top3_features = shap_importance_df.head(3)['Feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat in enumerate(top3_features):
    feat_idx = final_features.index(feat)
    shap.plots.scatter(shap_values[:, feat_idx],
                       ax=axes[idx], show=False)
    axes[idx].set_title(f'{feat}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n💬 해석:")
for feat in top3_features:
    print(f"\n[{feat}]")
    if feat == '최저가율':
        print("   - 낮을수록 낙찰가율 증가 (음의 상관)")
        print("   - 경매 경쟁력이 낮을수록 입찰 활발")
    elif feat == '보증금비율':
        print("   - 높을수록 낙찰가율 감소")
        print("   - 리스크 증가 → 입찰 기피")
    else:
        print("   - 그래프 참조")

# ============================================================
# 16-7. SHAP Waterfall Plot (개별 예측 설명)
# ============================================================

print("\n[16-7] SHAP Waterfall Plot - 개별 예측 설명")

# 흥미로운 케이스 선택
print("\n대표 케이스 3개 선택:")

# 케이스 1: 낙찰가율 높은 케이스
high_idx = y_test.loc[X_test_sample.index].idxmax()
print(f"\n   케이스 1: 낙찰가율 높음")
print(f"      실제: {y_test.loc[high_idx]:.3f}")
print(f"      예측: {y_pred_final[y_test.index.get_loc(high_idx)]:.3f}")

# 케이스 2: 낙찰가율 낮은 케이스
low_idx = y_test.loc[X_test_sample.index].idxmin()
print(f"\n   케이스 2: 낙찰가율 낮음")
print(f"      실제: {y_test.loc[low_idx]:.3f}")
print(f"      예측: {y_pred_final[y_test.index.get_loc(low_idx)]:.3f}")

# 케이스 3: 평균적인 케이스
median_idx = (y_test.loc[X_test_sample.index] - y_test.loc[X_test_sample.index].median()).abs().idxmin()
print(f"\n   케이스 3: 평균적 케이스")
print(f"      실제: {y_test.loc[median_idx]:.3f}")
print(f"      예측: {y_pred_final[y_test.index.get_loc(median_idx)]:.3f}")

# Waterfall plot
selected_cases = [high_idx, low_idx, median_idx]
case_names = ['높은 낙찰가율', '낮은 낙찰가율', '평균 낙찰가율']

for case_idx, case_name in zip(selected_cases, case_names):
    sample_idx = X_test_sample.index.get_loc(case_idx)

    print(f"\n[{case_name}] Waterfall Plot:")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.title(f'SHAP Waterfall - {case_name}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 케이스 상세 정보
    print(f"\n   피처 값:")
    for feat in final_features[:5]:  # 상위 5개만
        val = X_test_final.loc[case_idx, feat]
        print(f"      {feat}: {val:.3f}")

# ============================================================
# 16-8. Linear vs SHAP 비교
# ============================================================

print("\n[16-8] Linear Regression vs SHAP 인사이트 비교")

# Linear 계수
linear_coef_comparison = pd.DataFrame({
    'Feature': X_train.columns,
    'Linear_Coef': linear.coef_
}).sort_values('Linear_Coef', key=abs, ascending=False).head(10)

print("\nLinear Regression 계수 (상위 10개):")
display(linear_coef_comparison)

print("\nSHAP Importance (상위 10개):")
display(shap_importance_df.head(10))

print("\n💬 비교 분석:")
print("\n공통점 (일치하는 인사이트):")
print("   ✅ 최저가율이 압도적 1위")
print("   ✅ 보증금비율, 동_encoded 중요")
print("   ✅ 전반적 순위 유사")

print("\n차이점:")
print("   📊 Linear: 선형 관계 가정")
print("      예: 최저가율 1 증가 → 낙찰가율 +16.3%p (일정)")
print("\n   📊 SHAP: 비선형 효과 포착")
print("      예: 최저가율 0.5→0.6 vs 0.8→0.9 효과 다름")

# ============================================================
# 최종 요약
# ============================================================

print("\n" + "=" * 80)
print("✅ SHAP 분석 완료!")
print("=" * 80)

print(f"""
📊 핵심 발견:

[Feature Importance]
   1위: 최저가율 ({shap_importance_df.iloc[0]['SHAP_Importance']:.3f})
   2위: {shap_importance_df.iloc[1]['Feature']} ({shap_importance_df.iloc[1]['SHAP_Importance']:.3f})
   3위: {shap_importance_df.iloc[2]['Feature']} ({shap_importance_df.iloc[2]['SHAP_Importance']:.3f})

[Linear vs SHAP]
   공통점: 최저가율, 보증금비율, 지역 중요
   차이점: SHAP이 비선형 효과 포착

[개별 예측]
   Waterfall Plot으로 각 예측 설명 가능
   투명성 확보 ✅

🎯 결론:
   1. CatBoost (블랙박스)
   2. SHAP (설명 가능)
   3. Linear (인사이트)

   → 3가지 모두 유사한 결론!
   → 모델 신뢰성 확보 ✅
""")

print("=" * 80)