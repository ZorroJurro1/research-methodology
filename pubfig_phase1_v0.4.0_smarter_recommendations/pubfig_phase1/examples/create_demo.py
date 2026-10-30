"""Generate reproducible example figures using deterministic synthetic data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubfig import line_plot, bar_plot, scatter_plot, heatmap, save_publication_figure

OUT = Path(__file__).resolve().parents[1] / 'artifacts'
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261019)
epoch = np.arange(1, 31)
line = pd.DataFrame({'epoch':epoch,
    'score':0.55 + 0.38*(1-np.exp(-epoch/7)) + rng.normal(0,.012,len(epoch))})
fig, _ = line_plot(line, 'epoch', 'score', title='Illustrative model score by epoch', ylabel='Score')
print('line:', save_publication_figure(fig, 'line_demo', OUT)); plt.close(fig)
bar = pd.DataFrame({'method':['A','B','C','D'], 'time_ms':[23,18,21,15]})
fig, _ = bar_plot(bar, 'method', 'time_ms', title='Illustrative runtime comparison', ylabel='Runtime (ms)')
print('bar:', save_publication_figure(fig, 'bar_demo', OUT)); plt.close(fig)
scatter = pd.DataFrame({'feature':rng.normal(size=120)})
scatter['target'] = 1.8*scatter['feature'] + rng.normal(scale=.7, size=len(scatter))
fig, _ = scatter_plot(scatter, 'feature', 'target', title='Illustrative feature relationship')
print('scatter:', save_publication_figure(fig, 'scatter_demo', OUT)); plt.close(fig)
mat = pd.DataFrame(rng.normal(size=(8,5)), index=[f'row{i}' for i in range(1,9)], columns=[f'var{i}' for i in range(1,6)])
fig, _ = heatmap(mat, title='Illustrative matrix heatmap', annot=True)
print('heatmap:', save_publication_figure(fig, 'heatmap_demo', OUT)); plt.close(fig)
print(f'Wrote artifacts to {OUT}; values are synthetic, for software demonstration only.')
