# ============================================================
# 📊 전체 데이터 로드 및 텍스트 전처리
# ============================================================

print("=" * 80)
print("📊 데이터 로드 및 전처리")
print("=" * 80)

# CSV 파일들 로드
print("\n[1] CSV 파일 로드")

df_list = []

for csv_file in sorted(csv_files):
    try:
        df_temp = pd.read_csv(csv_file, encoding='utf-8-sig')

        # 연도 추출
        filename = os.path.basename(csv_file)
        if '2021' in filename:
            year = 2021
        elif '2022' in filename:
            year = 2022
        elif '2023' in filename:
            year = 2023
        elif '2024' in filename:
            year = 2024
        elif '2025' in filename:
            year = 2025
        else:
            year = 2023

        df_temp['연도'] = year
        df_list.append(df_temp)

        print(f"   ✅ {year}년: {len(df_temp):,}개")

    except Exception as e:
        print(f"   ⚠️ {os.path.basename(csv_file)}: {e}")

# 병합
df_full = pd.concat(df_list, ignore_index=True)

print(f"\n   전체 데이터: {len(df_full):,}개")
print(f"   컬럼 수: {len(df_full.columns)}개")

# ============================================================
# [2] 텍스트 컬럼 결합
# ============================================================

print("\n[2] 텍스트 데이터 전처리")

text_cols = ['매물특징', '부동산_현황']
available_text = [col for col in text_cols if col in df_full.columns]

print(f"\n   사용 가능한 텍스트 컬럼: {available_text}")

if len(available_text) > 0:

    def combine_text(row):
        """텍스트 컬럼들을 결합"""
        texts = []
        for col in available_text:
            if pd.notna(row.get(col)):
                text = str(row[col])
                # 길이 제한 (BERT 속도)
                if len(text) > 500:
                    text = text[:500]
                texts.append(text)
        return " ".join(texts) if texts else "정보 없음"

    print("   텍스트 결합 중...")
    df_full['combined_text'] = df_full.apply(combine_text, axis=1)

    # 통계
    lengths = df_full['combined_text'].str.len()
    print(f"\n   텍스트 길이 통계:")
    print(f"      평균: {lengths.mean():.0f}자")
    print(f"      중간값: {lengths.median():.0f}자")
    print(f"      최대: {lengths.max():.0f}자")

    # 샘플
    print(f"\n   📝 샘플 텍스트:")
    print("   " + "-" * 76)
    sample = df_full['combined_text'].iloc[0]
    print(f"   {sample[:150]}...")

    has_text = True

else:
    print("   ⚠️ 텍스트 컬럼 없음")
    df_full['combined_text'] = "정보 없음"
    has_text = False

print(f"\n✅ 전처리 완료!")
print(f"   텍스트 사용: {'가능' if has_text else '불가능'}")