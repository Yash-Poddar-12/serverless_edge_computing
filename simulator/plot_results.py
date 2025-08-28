import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics, out='metrics.png'):
    s = metrics.summary()
    print("Summary:", s)
    # simple bar chart of main metrics
    labels = ['successful', 'failed', 'cold_starts', 'avg_instances']
    values = [s['successful'], s['failed'], s['cold_starts'], s['avg_instances']]
    plt.figure(figsize=(8,4))
    plt.bar(labels, values)
    plt.tight_layout()
    plt.savefig(out)
    print("Saved plot to", out)
