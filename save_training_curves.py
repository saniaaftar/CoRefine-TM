import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

epochs = list(range(1, 51))
train_loss = [138.07,131.33,128.59,127.04,125.51,125.04,124.37,123.47,123.12,122.98,
              122.35,121.89,121.47,121.31,120.96,120.85,120.29,120.14,119.89,119.78,
              119.52,119.25,119.26,118.79,118.35,118.40,118.26,118.06,117.94,117.76,
              117.77,117.52,117.12,117.27,117.26,117.29,117.17,116.98,117.02,116.97,
              116.97,116.78,116.60,116.50,116.57,116.57,116.57,116.23,116.53,116.43]
triplet_loss = [0.1481,0.1483,0.1482,0.1483,0.1482,0.1483,0.1483,0.1482,0.1482,0.1483,
                0.1482,0.1481,0.1480,0.1481,0.1481,0.1482,0.1480,0.1480,0.1480,0.1481,
                0.1480,0.1480,0.1481,0.1480,0.1480,0.1480,0.1480,0.1480,0.1479,0.1479,
                0.1480,0.1480,0.1478,0.1479,0.1479,0.1479,0.1479,0.1478,0.1479,0.1479,
                0.1479,0.1478,0.1478,0.1477,0.1479,0.1478,0.1478,0.1477,0.1479,0.1478]

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(epochs, train_loss, color='#378ADD', linewidth=2, label='Training loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training loss', fontsize=12, color='#378ADD')
ax1.tick_params(axis='y', labelcolor='#378ADD')
ax1.set_ylim(110, 142)
ax1.grid(True, alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(epochs, triplet_loss, color='#D85A30', linewidth=1.5, 
         linestyle='--', label='Triplet loss')
ax2.set_ylabel('Triplet loss', fontsize=12, color='#D85A30')
ax2.tick_params(axis='y', labelcolor='#D85A30')
ax2.set_ylim(0.1470, 0.1495)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.title('Training convergence — Sahih al-Bukhari (K=30)', fontsize=12)
plt.tight_layout()
plt.savefig('fig_training_curves.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_training_curves.png', bbox_inches='tight', dpi=300)
print('Saved!')
