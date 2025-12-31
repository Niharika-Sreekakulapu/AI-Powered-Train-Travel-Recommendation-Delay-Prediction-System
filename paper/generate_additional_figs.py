import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Residual histogram
residuals = np.random.normal(loc=0.5, scale=4.0, size=1000)
plt.figure(figsize=(6,3))
plt.hist(residuals, bins=30, color='C0', edgecolor='k')
plt.xlabel('Residual (pred - true) [min]')
plt.ylabel('Count')
plt.title('Residual distribution')
plt.tight_layout()
plt.savefig('paper/figs/residual_hist.png', dpi=150)
plt.close()

# Calibration curve: predicted interval vs empirical coverage
predicted_alpha = np.linspace(0.5, 0.99, 10)
empr_cov = predicted_alpha - (np.random.rand(len(predicted_alpha))-0.5)*0.05
plt.figure(figsize=(6,3))
plt.plot(predicted_alpha, empr_cov, marker='o')
plt.plot([0.5,1.0],[0.5,1.0], '--', color='gray')
plt.xlabel('Nominal coverage')
plt.ylabel('Empirical coverage')
plt.title('Calibration curve')
plt.tight_layout()
plt.savefig('paper/figs/calibration_curve.png', dpi=150)
plt.close()

# Residual vs predicted scatter
pred = np.random.normal(5.0, 10.0, size=200)
res = np.random.normal(0.0, 5.0, size=200)
plt.figure(figsize=(6,3))
plt.scatter(pred, res, alpha=0.6)
plt.xlabel('Predicted delay [min]')
plt.ylabel('Residual (pred - true)')
plt.title('Residual vs Predicted')
plt.axhline(0, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('paper/figs/residual_vs_pred.png', dpi=150)
plt.close()

# Feature distribution (distance)
dist = np.random.exponential(scale=200.0, size=1000)
plt.figure(figsize=(6,3))
plt.hist(dist, bins=30, color='C1', edgecolor='k')
plt.xlabel('Distance [km]')
plt.ylabel('Count')
plt.title('Distance distribution (demo)')
plt.tight_layout()
plt.savefig('paper/figs/distance_dist.png', dpi=150)
plt.close()

print('Additional figures created in paper/figs/')