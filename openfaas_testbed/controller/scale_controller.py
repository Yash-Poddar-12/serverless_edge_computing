"""
Simple controller that scales a k8s deployment using kubectl.
Requires kubectl configured to your cluster.
"""
import subprocess
import time

def kubectl_scale(deployment, replicas, namespace='openfaas-fn'):
    cmd = ['kubectl', 'scale', 'deployment', deployment, f'--replicas={replicas}', '-n', namespace]
    subprocess.check_call(cmd)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', default='func-echo')
    parser.add_argument('--replicas', type=int, default=1)
    args = parser.parse_args()
    print("Scaling", args.deployment, "to", args.replicas)
    kubectl_scale(args.deployment, args.replicas)
