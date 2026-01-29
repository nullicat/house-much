# ============================================================
# 🎯 간단한 MLP 학습
# ============================================================

print("=" * 80)
print("🎯 간단한 MLP 학습")
print("=" * 80)

# 학습 함수 (간단 버전)
def train_simple_epoch(model, loader, criterion, optimizer, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# 평가 함수 (간단 버전)
def evaluate_simple(model, loader, device):
    """평가"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            output = model(X)

            predictions.extend(output.cpu().numpy())
            targets.extend(y.numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    return mae, rmse, r2, predictions

# ============================================================
# 학습 시작
# ============================================================

print("\n학습 시작!")

num_epochs = 100
best_simple_mae = float('inf')
patience_counter = 0
early_stop_patience = 15

simple_history = {
    'train_loss': [],
    'val_mae': [],
    'val_rmse': [],
    'val_r2': []
}

print(f"Epochs: {num_epochs}")
print(f"Early Stopping: {early_stop_patience}")
print("-" * 80)

for epoch in range(num_epochs):
    # 학습
    train_loss = train_simple_epoch(
        simple_model, simple_train_loader,
        simple_criterion, simple_optimizer, device
    )

    # 평가
    val_mae, val_rmse, val_r2, _ = evaluate_simple(
        simple_model, simple_test_loader, device
    )

    # 기록
    simple_history['train_loss'].append(train_loss)
    simple_history['val_mae'].append(val_mae)
    simple_history['val_rmse'].append(val_rmse)
    simple_history['val_r2'].append(val_r2)

    # 스케줄러
    simple_scheduler.step(val_mae)

    # 출력
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"MAE: {val_mae:.4f} | "
              f"RMSE: {val_rmse:.4f} | "
              f"R²: {val_r2:.4f}")

    # Best 모델 저장
    if val_mae < best_simple_mae:
        best_simple_mae = val_mae
        best_simple_epoch = epoch + 1
        patience_counter = 0

        torch.save(simple_model.state_dict(),
                   f'{backup_dir}/simple_mlp_best.pth')
    else:
        patience_counter += 1

    # Early Stopping
    if patience_counter >= early_stop_patience:
        print(f"\nEarly Stopping at Epoch {epoch+1}")
        break

print("\n" + "=" * 80)
print("✅ 학습 완료!")
print("=" * 80)

print(f"""
Best 성능:
- Epoch: {best_simple_epoch}
- MAE: {best_simple_mae:.4f}
""")

# ============================================================
# 최종 평가
# ============================================================

print("\n[최종 평가]")

# Best 모델 로드
simple_model.load_state_dict(
    torch.load(f'{backup_dir}/simple_mlp_best.pth')
)

# 평가
simple_mae, simple_rmse, simple_r2, simple_preds = evaluate_simple(
    simple_model, simple_test_loader, device
)

print(f"\n간단한 MLP 성능:")
print(f"   MAE:  {simple_mae:.4f}")
print(f"   RMSE: {simple_rmse:.4f}")
print(f"   R²:   {simple_r2:.4f}")

# ============================================================
# 전체 비교
# ============================================================

print("\n" + "=" * 80)
print("📊 전체 모델 비교")
print("=" * 80)

results_dl = pd.DataFrame({
    'Model': ['CatBoost', 'Multimodal', 'Simple MLP'],
    'MAE': [0.0747, final_mae, simple_mae],
    'RMSE': [0.1173, final_rmse, simple_rmse],
    'R²': [0.6710, final_r2, simple_r2],
    'Features': ['수치 10개', '수치 24개 + 텍스트', '수치 24개'],
    'Parameters': ['-', f'{total_params:,}', f'{simple_params:,}']
})

results_dl = results_dl.sort_values('MAE')

print("\n")
print(results_dl.to_string(index=False))

print("\n🏆 최고 성능:")
best_model = results_dl.iloc[0]
print(f"   모델: {best_model['Model']}")
print(f"   MAE: {best_model['MAE']:.4f}")

# Simple MLP vs CatBoost
improvement = (0.0747 - simple_mae) / 0.0747 * 100

print(f"\n[Simple MLP vs CatBoost]")
if simple_mae < 0.0747:
    print(f"   ✅ Simple MLP가 {improvement:.2f}% 더 좋음!")
elif simple_mae < 0.0750:
    print(f"   → 거의 비슷함 (차이 {abs(improvement):.2f}%)")
else:
    print(f"   → CatBoost가 여전히 우수")

print(f"\n모델 저장 완료:")
print(f"   {backup_dir}/simple_mlp_best.pth")