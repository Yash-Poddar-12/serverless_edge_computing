import matplotlib.pyplot as plt

def plot_metrics(metrics, out='metrics.png'):
    s = metrics.summary()
    print('Summary:', s)
    labels = ['successful', 'failed', 'cold_starts', 'avg_instances']
    values = [s['successful'], s['failed'], s['cold_starts'], s['avg_instances']]
    plt.figure(figsize=(8,4))
    plt.bar(labels, values)
    plt.title('Key metrics')
    plt.tight_layout()
    plt.savefig(out)
    print('Saved plot to', out)

    # latency CDF if available
    if hasattr(metrics, 'latencies') and metrics.latencies:
        plt.figure(figsize=(6,4))
        data = sorted(metrics.latencies)
        y = [i/len(data) for i in range(len(data))]
        plt.plot(data, y)
        plt.xlabel('Latency (s)')
        plt.ylabel('CDF')
        plt.title('Latency CDF')
        plt.tight_layout()
        plt.savefig('latency_cdf.png')
        print('Saved latency CDF to latency_cdf.png')
