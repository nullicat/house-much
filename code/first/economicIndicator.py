# ============================================================
# STEP 17: 경제 지표 추가 실험 (최종 수정)
# ============================================================

print("\n" + "=" * 80)
print("💰 STEP 17: 경제 지표 추가 (거시경제 효과 검증)")
print("=" * 80)

# ============================================================
# 17-0. df_featured 재생성 (인코딩 포함)
# ============================================================

print("\n[17-0] 피처 데이터 준비")

# df_sold에서 시작 (낙찰된 것만)
df_work = df_sold.copy()

# 타겟 생성
df_work['낙찰가율'] = df_work['낙찰가'] / df_work['감정가']

# 파생 변수들
df_work['최저가율'] = df_work['최저가'] / df_work['감정가']
df_work['보증금비율'] = df_work['보증금'] / df_work['감정가']
df_work['평당감정가'] = df_work['감정가'] / df_work['건물면적']
df_work['토지건물비율'] = df_work['건물면적'] / df_work['토지면적']
df_work['신건여부'] = (df_work['유찰횟수'] == 0).astype(int)
df_work['보증금유무'] = (df_work['보증금'] > 0).astype(int)
df_work['선순위초과'] = (df_work['보증금'] > df_work['감정가']).astype(int)

# 날짜 파싱
df_work['매각일_parsed'] = pd.to_datetime(df_work['매각일'], format='mixed')
df_work['매각_월'] = df_work['매각일_parsed'].dt.month
df_work['매각_분기'] = df_work['매각일_parsed'].dt.quarter

# Target Encoding
target_mean = df_work.groupby('구')['낙찰가율'].mean()
df_work['구_encoded'] = df_work['구'].map(target_mean)

target_mean_dong = df_work.groupby('동')['낙찰가율'].mean()
df_work['동_encoded'] = df_work['동'].map(target_mean_dong)

# 결측치 처리
df_work['구_encoded'] = df_work['구_encoded'].fillna(df_work['낙찰가율'].mean())
df_work['동_encoded'] = df_work['동_encoded'].fillna(df_work['낙찰가율'].mean())

print(f"✅ 피처 데이터 준비 완료: {len(df_work)}개")

# ============================================================
# 17-1. 경제 지표 병합
# ============================================================

print("\n[17-1] 경제 지표 병합")

# 매각_연월 생성
df_work['매각_연월'] = df_work['매각일_parsed'].dt.to_period('M')

# 경제 지표 간단 병합
economic_dir = '/content/economic_indicators'

# 기준금리
try:
    df_interest = pd.read_csv(f'{economic_dir}/01 기준금리.csv')
    df_interest['date'] = pd.to_datetime(df_interest['date'])
    df_interest['연월'] = df_interest['date'].dt.to_period('M')
    df_interest_agg = df_interest.groupby('연월')['기준금리'].mean().reset_index()

    df_work = df_work.merge(df_interest_agg, left_on='매각_연월', right_on='연월', how='left')
    print(f"   ✅ 기준금리 병합 완료: {df_work['기준금리'].notna().sum()}개")
except Exception as e:
    print(f"   ⚠️ 기준금리 실패: {e}")

# 주택담보대출금리
try:
    df_mortgage = pd.read_csv(f'{economic_dir}/02 변동형주택담보대출금리.csv')
    df_mortgage['date'] = pd.to_datetime(df_mortgage['date'])
    df_mortgage['연월'] = df_mortgage['date'].dt.to_period('M')
    df_mortgage_agg = df_mortgage.groupby('연월')['변동형주택담보대출금리'].mean().reset_index()

    df_work = df_work.merge(df_mortgage_agg, left_on='매각_연월', right_on='연월', how='left', suffixes=('', '_mortgage'))
    print(f"   ✅ 주택담보대출금리 병합 완료: {df_work['변동형주택담보대출금리'].notna().sum()}개")
except Exception as e:
    print(f"   ⚠️ 주택담보대출금리 실패: {e}")

# 전세자금대출금리
try:
    df_jeonse = pd.read_csv(f'{economic_dir}/03 전세자금대출금리.csv')
    df_jeonse['date'] = pd.to_datetime(df_jeonse['date'])
    df_jeonse['연월'] = df_jeonse['date'].dt.to_period('M')
    df_jeonse_agg = df_jeonse.groupby('연월')['전세자금대출금리'].mean().reset_index()

    df_work = df_work.merge(df_jeonse_agg, left_on='매각_연월', right_on='연월', how='left', suffixes=('', '_jeonse'))
    print(f"   ✅ 전세자금대출금리 병합 완료: {df_work['전세자금대출금리'].notna().sum()}개")
except Exception as e:
    print(f"   ⚠️ 전세자금대출금리 실패: {e}")

# 결측치 처리
economic_cols = ['기준금리', '변동형주택담보대출금리', '전세자금대출금리']
available_econ = []

for col in economic_cols:
    if col in df_work.columns:
        # 전방향/후방향 채우기
        df_work[col] = df_work.groupby('연도')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        # 남은 결측치는 평균
        df_work[col] = df_work[col].fillna(df_work[col].mean())

        non_null = df_work[col].notna().sum()
        if non_null > len(df_work) * 0.5:
            available_econ.append(col)

print(f"\n✅ 사용 가능한 경제 지표: {len(available_econ)}개")

# ============================================================
# 17-2. Train/Test 분할
# ============================================================

print("\n[17-2] Train/Test 분할")

# 기본 10개 피처
base_features = ['최저가율', '최저가', '신건여부', '보증금비율',
                 '동_encoded', '평당감정가', '감정가', '건물면적',
                 '구_encoded', '매각_월']

# 전체 피처
features_with_econ = base_features + available_econ

print(f"   기본 피처: {len(base_features)}개")
print(f"   경제 지표: {len(available_econ)}개")
print(f"   총 피처: {len(features_with_econ)}개")

# 분할
df_train_econ = df_work[df_work['연도'] < 2025]
df_test_econ = df_work[df_work['연도'] == 2025]

X_train_base = df_train_econ[base_features]
X_train_econ = df_train_econ[features_with_econ]
y_train_econ = df_train_econ['낙찰가율']

X_test_base = df_test_econ[base_features]
X_test_econ = df_test_econ[features_with_econ]
y_test_econ = df_test_econ['낙찰가율']

print(f"   Train: {len(X_train_econ)}개")
print(f"   Test: {len(X_test_econ)}개")

# ============================================================
# 17-3. 모델 비교
# ============================================================

print("\n[17-3] 모델 비교")

# 모델 1: 기본만
print("\n[모델 1] 기본 피처만")
model_base = CatBoostRegressor(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    random_state=42,
    verbose=0
)

model_base.fit(X_train_base, y_train_econ)
y_pred_base = model_base.predict(X_test_base)

mae_base = mean_absolute_error(y_test_econ, y_pred_base)
r2_base = r2_score(y_test_econ, y_pred_base)

print(f"   MAE: {mae_base:.4f}")
print(f"   R²:  {r2_base:.4f}")

# 모델 2: 기본 + 경제
print(f"\n[모델 2] 기본 + 경제 지표 ({len(features_with_econ)}개)")
model_econ = CatBoostRegressor(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    random_state=42,
    verbose=0
)

model_econ.fit(X_train_econ, y_train_econ)
y_pred_econ = model_econ.predict(X_test_econ)

mae_econ = mean_absolute_error(y_test_econ, y_pred_econ)
r2_econ = r2_score(y_test_econ, y_pred_econ)

print(f"   MAE: {mae_econ:.4f}")
print(f"   R²:  {r2_econ:.4f}")

# 비교
improvement = (mae_base - mae_econ) / mae_base * 100

print(f"\n[비교]")
print(f"   MAE 변화: {mae_econ - mae_base:+.4f} ({improvement:+.2f}%)")
print(f"   R² 변화: {r2_econ - r2_base:+.4f}")

# 통계 검정
from scipy.stats import ttest_rel

residuals_base = np.abs(y_test_econ - y_pred_base)
residuals_econ = np.abs(y_test_econ - y_pred_econ)

t_stat, p_value = ttest_rel(residuals_base, residuals_econ)

print(f"\n[통계 검정]")
print(f"   t-통계량: {t_stat:.3f}")
print(f"   p-value: {p_value:.4f}")

if p_value < 0.05:
    better = "경제 지표" if mae_econ < mae_base else "기본만"
    print(f"   → ✅ {better}가 통계적으로 유의하게 우수")
else:
    print(f"   → 통계적으로 유의한 차이 없음")

# Feature Importance
print(f"\n[경제 지표 중요도]")

importance_all = model_econ.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': features_with_econ,
    'Importance': importance_all
}).sort_values('Importance', ascending=False)

econ_importance = importance_df[importance_df['Feature'].isin(available_econ)]

print("\n경제 지표 Feature Importance:")
display(econ_importance)

print(f"\n전체 중 경제 지표 순위:")
for idx, row in econ_importance.iterrows():
    feat = row['Feature']
    rank = importance_df[importance_df['Feature'] == feat].index[0] + 1
    print(f"   {feat}: {rank}위 / {len(features_with_econ)}개")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance
ax1 = axes[0]
top15 = importance_df.head(15)
colors = ['coral' if f in available_econ else 'steelblue' for f in top15['Feature']]
ax1.barh(range(len(top15)), top15['Importance'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top15)))
ax1.set_yticklabels(top15['Feature'])
ax1.set_xlabel('Feature Importance', fontsize=12)
ax1.set_title('Feature Importance (빨강=경제지표)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(alpha=0.3, axis='x')

# MAE 비교
ax2 = axes[1]
models = ['기본 10개', f'기본+경제 {len(features_with_econ)}개']
maes = [mae_base, mae_econ]
colors_bar = ['steelblue', 'coral']
bars = ax2.bar(models, maes, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('경제 지표 추가 효과', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

for bar, val in zip(bars, maes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

# ============================================================
# 최종 요약
# ============================================================

print("\n" + "=" * 80)
print("✅ 경제 지표 실험 완료!")
print("=" * 80)

print(f"""
📊 H4 가설 검증 결과:

[가설]
   거시경제 지표가 낙찰가율에 영향을 미친다

[실험]
   기본 {len(base_features)}개 vs 기본+경제 {len(features_with_econ)}개
   경제 지표: {', '.join(available_econ)}

[결과]
   기본만: MAE {mae_base:.4f}, R² {r2_base:.4f}
   +경제:  MAE {mae_econ:.4f}, R² {r2_econ:.4f}
   개선율: {improvement:+.2f}%
   p-value: {p_value:.4f}

[경제 지표 중요도]
""")

for idx, row in econ_importance.iterrows():
    rank = importance_df[importance_df['Feature'] == row['Feature']].index[0] + 1
    print(f"   {row['Feature']}: {rank}위 ({row['Importance']:.1f}점)")

print(f"\n[결론]")

if improvement > 1 and p_value < 0.05:
    print(f"   ✅ H4 채택: 경제 지표가 예측 성능 개선")
    print(f"   거시경제가 낙찰가율에 유의미한 영향")
elif improvement > 0 and improvement <= 1:
    print(f"   △ H4 부분 채택: 개선되나 미미함 ({improvement:.1f}%)")
    print(f"   통계적 유의: {'있음' if p_value < 0.05 else '없음'}")
    print(f"   실무적 가치: 낮음")
else:
    print(f"   ❌ H4 기각: 경제 지표 효과 없음")
    print(f"   물건 고유 특성이 훨씬 중요")

print("=" * 80)