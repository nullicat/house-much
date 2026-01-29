# ============================================================
# 이상치 제거
# ============================================================

print("=" * 80)
print("🧹 이상치 제거")
print("=" * 80)

print(f"\n제거 전: {len(df_sold):,}개")

# 제거할 이상치 확인
outliers_to_remove = df_sold[df_sold['낙찰가율'] > 2.0]
print(f"\n제거 대상: {len(outliers_to_remove):,}개 (낙찰가율 > 2.0)")
print(f"   - 비율: {len(outliers_to_remove)/len(df_sold)*100:.3f}%")

# 이상치 통계
print(f"\n제거 대상 통계:")
print(f"   - 최소 낙찰가율: {outliers_to_remove['낙찰가율'].min():.2f}")
print(f"   - 최대 낙찰가율: {outliers_to_remove['낙찰가율'].max():.2f}")
print(f"   - 평균 낙찰가율: {outliers_to_remove['낙찰가율'].mean():.2f}")

# 제거
df_clean = df_sold[df_sold['낙찰가율'] <= 2.0].copy()

print(f"\n제거 후: {len(df_clean):,}개")
print(f"   - 제거율: {len(outliers_to_remove)/len(df_sold)*100:.2f}%")
print(f"   - 보존율: {len(df_clean)/len(df_sold)*100:.2f}%")

# 제거 후 통계
print(f"\n제거 후 낙찰가율 통계:")
print(f"   - 최소: {df_clean['낙찰가율'].min():.3f}")
print(f"   - 평균: {df_clean['낙찰가율'].mean():.3f}")
print(f"   - 중앙: {df_clean['낙찰가율'].median():.3f}")
print(f"   - 최대: {df_clean['낙찰가율'].max():.3f}")
print(f"   - 표준편차: {df_clean['낙찰가율'].std():.3f}")

# 시각화 비교
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 제거 전
axes[0].hist(df_sold['낙찰가율'], bins=50, edgecolor='black', alpha=0.7, color='red')
axes[0].axvline(2.0, color='blue', linestyle='--', linewidth=2, label='제거 기준: 2.0')
axes[0].set_xlabel('낙찰가율')
axes[0].set_ylabel('빈도')
axes[0].set_title(f'제거 전 ({len(df_sold):,}개)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 제거 후
axes[1].hist(df_clean['낙찰가율'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(df_clean['낙찰가율'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'평균: {df_clean["낙찰가율"].mean():.3f}')
axes[1].set_xlabel('낙찰가율')
axes[1].set_ylabel('빈도')
axes[1].set_title(f'제거 후 ({len(df_clean):,}개)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("✅ 이상치 제거 완료!")
print("=" * 80)

# df_clean을 이제 주 데이터로 사용
df_sold = df_clean.copy()

print(f"\n최종 분석 데이터: {len(df_sold):,}개")