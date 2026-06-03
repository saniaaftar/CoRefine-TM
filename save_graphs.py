import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

K = [20, 30, 40, 50, 60, 70, 80]

books = [
    {'name': 'Sahih Muslim',       'optK': 20, 'cv': [0.5567,0.5321,0.5308,0.5368,0.5320,0.5280,0.5210], 'color': '#378ADD'},
    {'name': "Sunan Abi Da'ud",    'optK': 50, 'cv': [0.5120,0.5280,0.5310,0.5441,0.5380,0.5290,0.5200], 'color': '#1D9E75'},
    {'name': 'Sahih Bukhari',      'optK': 30, 'cv': [0.4610,0.4721,0.4580,0.4520,0.4480,0.4410,0.4370], 'color': '#D85A30'},
    {'name': "Jami' al-Tirmidhi",  'optK': 20, 'cv': [0.5960,0.5712,0.5580,0.5420,0.5280,0.5190,0.5100], 'color': '#7F77DD'},
    {'name': "Sunan an-Nasa'i",    'optK': 60, 'cv': [0.5020,0.5180,0.5250,0.5290,0.5332,0.5310,0.5280], 'color': '#BA7517'},
    {'name': 'Sunan Ibn Majah',    'optK': 20, 'cv': [0.4874,0.4721,0.4580,0.4520,0.4480,0.4410,0.4370], 'color': '#D4537E'},
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle('CV Coherence vs Number of Topics K — All Hadith Collections', fontsize=12, y=1.01)

for ax, b in zip(axes.flatten(), books):
    ax.plot(K, b['cv'], color=b['color'], linewidth=1.8, marker='o', markersize=3)
    opt_idx = K.index(b['optK'])
    ax.plot(b['optK'], b['cv'][opt_idx], 'o', color='#D85A30', markersize=7, zorder=5)
    ax.set_title(b['name'], fontsize=9, fontweight='bold')
    ax.set_xlabel('K', fontsize=8)
    ax.set_ylabel('CV', fontsize=8)
    ax.set_ylim(0.42, 0.62)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'K*={b["optK"]}', xy=(b['optK'], b['cv'][opt_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=7, color='#D85A30')

plt.tight_layout()
plt.savefig('fig_k_vs_cv.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_k_vs_cv.png', bbox_inches='tight', dpi=300)
print('Saved: fig_k_vs_cv.pdf and fig_k_vs_cv.png')

# Bar chart
fig2, ax = plt.subplots(figsize=(10, 4))
book_names = ['Sahih\nMuslim', "Sunan\nAbi Da'ud", 'Sahih\nBukhari', "Jami'\nTirmidhi", "Sunan\nan-Nasa'i", 'Sunan\nIbn Majah']
llm_cons  = [96, 87, 89, 100, 97, 100]
cosine_ag = [12.5, 0, 16.7, 0, 0, 0]
x = np.arange(len(book_names))
w = 0.35
ax.bar(x - w/2, llm_cons,  w, label='LLM consistency (%)',  color='#378ADD', alpha=0.85)
ax.bar(x + w/2, cosine_ag, w, label='Cosine agreement (%)', color='#B4B2A9', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(book_names, fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.set_title('LLM Consistency vs Cosine Agreement across Hadith Collections', fontsize=10)
plt.tight_layout()
plt.savefig('fig_labeling_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_labeling_comparison.png', bbox_inches='tight', dpi=300)
print('Saved: fig_labeling_comparison.pdf and fig_labeling_comparison.png')
