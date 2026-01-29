# ============================================================
# 🧠 PyTorch Multimodal 모델 구축
# ============================================================

print("=" * 80)
print("🧠 PyTorch Multimodal 모델 구축")
print("=" * 80)

print("""
💡 모델 구조:

Branch 1: 수치 (24개 피처)
   Input(24) → FC(64) → ReLU → Dropout
   → FC(32) → ReLU

Branch 2: 텍스트 (768차원 BERT)
   Input(768) → FC(256) → ReLU → Dropout
   → FC(128) → ReLU → Dropout
   → FC(32) → ReLU

Fusion:
   Concat(32 + 32 = 64)
   → FC(32) → ReLU
   → FC(1) → Output
""")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# [1] Dataset 클래스
# ============================================================

print("\n[1] Dataset 클래스 정의")

class AuctionDataset(Dataset):
    """경매 데이터셋"""

    def __init__(self, X_numeric, X_bert, y):
        self.X_numeric = torch.FloatTensor(X_numeric)
        self.X_bert = torch.FloatTensor(X_bert)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'numeric': self.X_numeric[idx],
            'bert': self.X_bert[idx],
            'target': self.y[idx]
        }

# Dataset 생성
train_dataset = AuctionDataset(X_train_numeric, X_train_bert, y_train_dl)
test_dataset = AuctionDataset(X_test_numeric, X_test_bert, y_test_dl)

print(f"   Train Dataset: {len(train_dataset)}개")
print(f"   Test Dataset: {len(test_dataset)}개")

# DataLoader 생성
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# ============================================================
# [2] 모델 클래스
# ============================================================

print("\n[2] Multimodal 모델 정의")

class MultimodalModel(nn.Module):
    """수치 + 텍스트 Multimodal 모델"""

    def __init__(self, numeric_dim=24, bert_dim=768):
        super().__init__()

        # Branch 1: 수치 피처
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Branch 2: BERT 임베딩
        self.bert_branch = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),  # 32 + 32 = 64
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, numeric, bert):
        # 각 Branch 통과
        numeric_out = self.numeric_branch(numeric)
        bert_out = self.bert_branch(bert)

        # 결합
        combined = torch.cat([numeric_out, bert_out], dim=1)

        # 최종 예측
        output = self.fusion(combined)

        return output.squeeze()

# 모델 생성
model = MultimodalModel(
    numeric_dim=X_train_numeric.shape[1],
    bert_dim=X_train_bert.shape[1]
)

model = model.to(device)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   ✅ 모델 생성 완료")
print(f"   총 파라미터: {total_params:,}개")
print(f"   학습 가능: {trainable_params:,}개")

# ============================================================
# [3] 손실 함수 및 옵티마이저
# ============================================================

print("\n[3] 학습 설정")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

print(f"   손실 함수: MSE")
print(f"   옵티마이저: Adam")
print(f"   학습률: 0.001")
print(f"   스케줄러: ReduceLROnPlateau")

# ============================================================
# 완료
# ============================================================

print("\n" + "=" * 80)
print("✅ 모델 구축 완료!")
print("=" * 80)

print(f"""
준비된 것:
- 모델: Multimodal ({total_params:,} 파라미터)
- 데이터: Train {len(train_dataset):,}개, Test {len(test_dataset):,}개
- 디바이스: {device}

다음 단계: 학습 시작!
""")