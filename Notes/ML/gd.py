"""
Gradient Descent - Linear Regression
Feature: Car weight (1000s of pounds)
Label:   Miles per gallon (MPG)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Data ─────────────────────────────────────────────────────────────────────
X_raw = np.array([3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37])
y     = np.array([18,  15,   18,   16,   15,   14,   24 ], dtype=float)
n     = len(X_raw)

# Feature normalisation (z-score) – makes gradient descent better-behaved
X_mean, X_std = X_raw.mean(), X_raw.std()
X = (X_raw - X_mean) / X_std          # normalised feature

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(X, w, b):
    return w * X + b

def mse_loss(X, y, w, b):
    return np.mean((predict(X, w, b) - y) ** 2)

def gradients(X, y, w, b):
    err  = predict(X, w, b) - y
    dw   = (2 / n) * np.dot(X, err)
    db   = (2 / n) * err.sum()
    return dw, db

# ── Gradient Descent ──────────────────────────────────────────────────────────
lr      = 0.1
epochs  = 200
w, b    = 0.0, 0.0          # initialise at origin

loss_history = []
w_history    = []
b_history    = []

print("=" * 55)
print(f"{'Gradient Descent – Linear Regression':^55}")
print("=" * 55)
print(f"  Samples : {n}")
print(f"  LR (η)  : {lr}")
print(f"  Epochs  : {epochs}")
print("-" * 55)
print(f"{'Epoch':>6}  {'Loss (MSE)':>12}  {'w':>10}  {'b':>10}")
print("-" * 55)

for epoch in range(1, epochs + 1):
    loss = mse_loss(X, y, w, b)
    loss_history.append(loss)
    w_history.append(w)
    b_history.append(b)

    dw, db = gradients(X, y, w, b)
    w -= lr * dw
    b -= lr * db

    if epoch in (1, 5, 10, 25, 50, 100, 150, 200):
        print(f"{epoch:>6}  {loss:>12.5f}  {w:>10.5f}  {b:>10.5f}")

print("-" * 55)
final_loss = mse_loss(X, y, w, b)
print(f"\n✓ Final  w = {w:.5f}  (normalised space)")
print(f"✓ Final  b = {b:.5f}")
print(f"✓ Final MSE Loss = {final_loss:.5f}")

# Convert w back to original feature space for display
w_orig = w / X_std
b_orig = b - w * X_mean / X_std
print(f"\n  Regression line (original scale):")
print(f"  MPG = {w_orig:.4f} × Weight + {b_orig:.4f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor('#F8F8F6')
gs  = gridspec.GridSpec(1, 2, wspace=0.38)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

TEAL  = '#0F6E56'
BLUE  = '#185FA5'
CORAL = '#D85A30'
AMBER = '#BA7517'
GRAY  = '#5F5E5A'

# ── Panel 1: Loss curve ───────────────────────────────────────────────────────
ax1.set_facecolor('#F8F8F6')
ax1.plot(range(1, epochs + 1), loss_history, color=TEAL, lw=2, label='MSE loss')

# Mark key epochs
marks = [1, 10, 50, 100, 200]
for ep in marks:
    lv = loss_history[ep - 1]
    ax1.scatter(ep, lv, color=CORAL, s=55, zorder=5)
    ax1.annotate(f'ep {ep}\n{lv:.2f}',
                 xy=(ep, lv), xytext=(ep + 6, lv + 0.3),
                 fontsize=7.5, color=CORAL,
                 arrowprops=dict(arrowstyle='->', color=CORAL, lw=0.8))

ax1.set_title('Loss over Training Epochs', fontsize=13, fontweight='500', color=GRAY, pad=10)
ax1.set_xlabel('Epoch', fontsize=11, color=GRAY)
ax1.set_ylabel('MSE Loss', fontsize=11, color=GRAY)
ax1.tick_params(colors=GRAY, labelsize=9)
for spine in ax1.spines.values():
    spine.set_edgecolor('#C4C3BB')
    spine.set_linewidth(0.6)
ax1.grid(axis='y', linestyle='--', linewidth=0.5, color='#D3D1C7', alpha=0.8)
ax1.set_xlim(0, epochs + 5)

# ── Panel 2: Final regression line ───────────────────────────────────────────
ax2.set_facecolor('#F8F8F6')

# Scatter – data points
ax2.scatter(X_raw, y, color=BLUE, s=80, zorder=5, label='Data points', edgecolors='white', linewidths=0.8)

# Regression line in original feature space
x_line = np.linspace(X_raw.min() - 0.2, X_raw.max() + 0.2, 200)
y_line = w_orig * x_line + b_orig
ax2.plot(x_line, y_line, color=TEAL, lw=2.2, label=f'Fit: MPG = {w_orig:.2f}·W + {b_orig:.2f}')

# Residual lines
y_pred_orig = w_orig * X_raw + b_orig
for xi, yi, yp in zip(X_raw, y, y_pred_orig):
    ax2.plot([xi, xi], [yi, yp], color=CORAL, lw=0.9, linestyle='--', alpha=0.65)

ax2.set_title('Final Regression Line', fontsize=13, fontweight='500', color=GRAY, pad=10)
ax2.set_xlabel('Weight (1000s lbs)', fontsize=11, color=GRAY)
ax2.set_ylabel('Miles per Gallon', fontsize=11, color=GRAY)
ax2.tick_params(colors=GRAY, labelsize=9)
for spine in ax2.spines.values():
    spine.set_edgecolor('#C4C3BB')
    spine.set_linewidth(0.6)
ax2.grid(linestyle='--', linewidth=0.5, color='#D3D1C7', alpha=0.8)
ax2.legend(fontsize=8.5, framealpha=0.5, edgecolor='#C4C3BB')

# Annotate residuals label
ax2.annotate('Residuals', xy=(3.44, 17), xytext=(3.8, 20.5),
             fontsize=8, color=CORAL,
             arrowprops=dict(arrowstyle='->', color=CORAL, lw=0.7))

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle('Gradient Descent — Car Weight vs MPG', fontsize=14,
             fontweight='500', color=GRAY, y=1.01)

plt.savefig('/mnt/user-data/outputs/gradient_descent_regression.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("\n✓ Plot saved.")
plt.close()
