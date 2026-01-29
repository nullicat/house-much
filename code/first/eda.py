# ============================================================
# STEP 6: 탐색적 데이터 분석 (EDA)
# ============================================================

print("\n" + "=" * 80)
print("📊 STEP 6: 탐색적 데이터 분석 (EDA)")
print("=" * 80)

# 6-1. 타겟 변수 생성 (낙찰가율)
print("\n[6-1] 타겟 변수 생성: 낙찰가율")

# 낙찰된 데이터만
df_sold = df_all[df_all['낙찰가'].notna()].copy()

# 낙찰가율 = 낙찰가 / 감정가
df_sold['낙찰가율'] = (df_sold['낙찰가'] / df_sold['감정가'])

print(f"   - 낙찰가율 범위: {df_sold['낙찰가율'].min():.2f} ~ {df_sold['낙찰가율'].max():.2f}")
print(f"   - 낙찰가율 평균: {df_sold['낙찰가율'].mean():.2f} ({df_sold['낙찰가율'].mean()*100:.1f}%)")
print(f"   - 낙찰가율 중앙: {df_sold['낙찰가율'].median():.2f} ({df_sold['낙찰가율'].median()*100:.1f}%)")

# 6-2. 낙찰가율 분포 시각화
print("\n[6-2] 낙찰가율 분포 시각화")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 히스토그램
axes[0].hist(df_sold['낙찰가율'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(df_sold['낙찰가율'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'평균: {df_sold["낙찰가율"].mean():.2f}')
axes[0].axvline(df_sold['낙찰가율'].median(), color='blue',
                linestyle='--', linewidth=2, label=f'중앙: {df_sold["낙찰가율"].median():.2f}')
axes[0].set_xlabel('낙찰가율 (낙찰가/감정가)')
axes[0].set_ylabel('빈도')
axes[0].set_title('낙찰가율 분포 (히스토그램)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 박스플롯
axes[1].boxplot(df_sold['낙찰가율'], vert=True)
axes[1].set_ylabel('낙찰가율')
axes[1].set_title('낙찰가율 분포 (박스플롯)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 6-3. 유찰횟수별 낙찰가율
print("\n[6-3] 유찰횟수별 낙찰가율 분석")

auction_analysis = df_sold.groupby('유찰횟수').agg({
    '낙찰가율': ['count', 'mean', 'std', 'min', 'max']
}).round(3)

print(auction_analysis)

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df_sold.groupby('유찰횟수')['낙찰가율'].mean().plot(kind='bar', color='steelblue')
plt.xlabel('유찰횟수')
plt.ylabel('평균 낙찰가율')
plt.title('유찰횟수별 평균 낙찰가율')
plt.xticks(rotation=0)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
df_sold.boxplot(column='낙찰가율', by='유찰횟수', figsize=(12, 5))
plt.xlabel('유찰횟수')
plt.ylabel('낙찰가율')
plt.title('유찰횟수별 낙찰가율 분포')
plt.suptitle('')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 6-4. 지역(구)별 낙찰가율
print("\n[6-4] 지역(구)별 낙찰가율 분석")

gu_analysis = df_sold.groupby('구').agg({
    '낙찰가율': ['count', 'mean']
}).round(3)
gu_analysis.columns = ['건수', '평균_낙찰가율']
gu_analysis = gu_analysis.sort_values('평균_낙찰가율', ascending=False)

print("\n상위 10개 구:")
print(gu_analysis.head(10))

print("\n하위 10개 구:")
print(gu_analysis.tail(10))

# 시각화
plt.figure(figsize=(14, 6))
gu_analysis['평균_낙찰가율'].plot(kind='bar', color='coral')
plt.xlabel('구')
plt.ylabel('평균 낙찰가율')
plt.title('구별 평균 낙찰가율')
plt.xticks(rotation=45, ha='right')
plt.axhline(df_sold['낙찰가율'].mean(), color='red',
            linestyle='--', label=f'전체 평균: {df_sold["낙찰가율"].mean():.2f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6-5. 용도별 낙찰가율
print("\n[6-5] 용도별 낙찰가율 분석")

usage_analysis = df_sold.groupby('용도').agg({
    '낙찰가율': ['count', 'mean', 'std']
}).round(3)

print(usage_analysis)

# 시각화
plt.figure(figsize=(10, 6))
df_sold.boxplot(column='낙찰가율', by='용도', figsize=(10, 6))
plt.xlabel('용도')
plt.ylabel('낙찰가율')
plt.title('용도별 낙찰가율 분포')
plt.suptitle('')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6-6. 보증금 영향 분석
print("\n[6-6] 보증금 영향 분석")

# 보증금비율 생성
df_sold['보증금비율'] = df_sold['보증금'] / df_sold['감정가']

# 보증금 유무별 비교
with_deposit = df_sold[df_sold['보증금'].notna() & (df_sold['보증금'] > 0)]
without_deposit = df_sold[df_sold['보증금'].isna() | (df_sold['보증금'] == 0)]

print(f"\n보증금 있음: {len(with_deposit):,}개")
print(f"   - 평균 낙찰가율: {with_deposit['낙찰가율'].mean():.3f}")
print(f"\n보증금 없음: {len(without_deposit):,}개")
print(f"   - 평균 낙찰가율: {without_deposit['낙찰가율'].mean():.3f}")

# t-test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(with_deposit['낙찰가율'],
                             without_deposit['낙찰가율'])
print(f"\nt-test 결과:")
print(f"   - t-통계량: {t_stat:.3f}")
print(f"   - p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"   → 보증금 유무에 따른 낙찰가율 차이 통계적으로 유의미! ✅")

# 6-7. 연도별 트렌드
print("\n[6-7] 연도별 낙찰가율 트렌드")

year_trend = df_sold.groupby('연도').agg({
    '낙찰가율': ['count', 'mean', 'std']
}).round(3)

print(year_trend)

plt.figure(figsize=(10, 5))
df_sold.boxplot(column='낙찰가율', by='연도', figsize=(10, 5))
plt.xlabel('연도')
plt.ylabel('낙찰가율')
plt.title('연도별 낙찰가율 분포')
plt.suptitle('')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("✅ EDA 완료!")
print("=" * 80)