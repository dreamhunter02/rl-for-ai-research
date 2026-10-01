import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
rows = [r for r in json.loads((root / 'trainer_state.json').read_text())['log_history'] if 'loss' in r]
x = [r['step'] for r in rows]
y = [r['loss'] for r in rows]
smooth = [sum(y[i-9:i+1])/10 for i in range(9, len(y))]
fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
ax.plot(x, y, color='#a9bdd3', linewidth=1, alpha=.8, label='Per-step training loss')
ax.plot(x[9:], smooth, color='#1762a4', linewidth=2.5, label='10-step trailing average')
ax.axvline(71, color='#888888', linestyle='--', linewidth=1, label='End of epoch 1')
ax.set(title='Qwen3.5-4B · FinanceBench QLoRA SFT', xlabel='Optimizer step', ylabel='Assistant-token training loss', xlim=(1,142), ylim=(0,None))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=.18)
ax.legend(frameon=False, fontsize=9)
fig.savefig(root / 'training_loss.png', dpi=180)
fig.savefig(root / 'training_loss.pdf')
print({'steps':len(rows),'first10_mean':sum(y[:10])/10,'last10_mean':sum(y[-10:])/10})
