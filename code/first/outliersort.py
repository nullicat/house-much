# ============================================================
# 이상치 데이터 확인
# ============================================================

print("=" * 80)
print("🔍 이상치 데이터 확인: 낙찰가율 > 2인 케이스")
print("=" * 80)

# 낙찰가율 > 2 (200% 이상) 데이터 추출
outliers = df_sold[df_sold['낙찰가율'] > 2.0].copy()

print(f"\n총 {len(outliers)}개 발견")

# 상세 정보 출력
print("\n상세 정보:")
display_cols = ['연도', '사건번호', 'm_code', '구', '동', '용도',
                '감정가', '최저가', '낙찰가', '낙찰가율', '유찰횟수', '매각일']

for idx, row in outliers.iterrows():
    print("\n" + "-" * 80)
    for col in display_cols:
        if col in row:
            if col in ['감정가', '최저가', '낙찰가']:
                print(f"{col}: {row[col]:,.0f}원")
            elif col == '낙찰가율':
                print(f"{col}: {row[col]:.2f} ({row[col]*100:.1f}%)")
            else:
                print(f"{col}: {row[col]}")

    # 파일 위치
    year = row['연도']
    print(f"\n📁 파일 위치: /content/pre/{year}_static_pre_0123_0354.csv")
    print(f"📝 m_code로 검색: {row['m_code']}")

# CSV로 저장 (자세히 보기)
outliers[display_cols].to_csv('/content/outliers_check.csv', index=False, encoding='utf-8-sig')
print("\n" + "=" * 80)
print("✅ 이상치 데이터를 /content/outliers_check.csv에 저장했습니다.")
print("=" * 80)

# 다운로드
from google.colab import files
files.download('/content/outliers_check.csv')