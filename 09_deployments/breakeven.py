"""Serverless vs dedicated break-even calculator.

All prices are INPUTS. Look up current numbers at https://fireworks.ai/pricing.
Get tokens_per_gpu_hour by BENCHMARKING your workload (07_production/benchmark.py).

Usage example:
  python 09_deployments/breakeven.py --in-price 0.9 --out-price 0.9 --gpu-hr 6.0 \
      --gpus 1 --tokens-per-gpu-hour 5e6 --output-share 0.25
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in-price", type=float, required=True, help="serverless $/1M input tokens")
ap.add_argument("--out-price", type=float, required=True, help="serverless $/1M output tokens")
ap.add_argument("--output-share", type=float, default=0.25, help="fraction of tokens that are output")
ap.add_argument("--gpu-hr", type=float, required=True, help="dedicated $/GPU-hour")
ap.add_argument("--gpus", type=int, default=1, help="GPUs per replica")
ap.add_argument("--tokens-per-gpu-hour", type=float, required=True, help="measured total tokens one replica handles per hour")
a = ap.parse_args()

blended = (1 - a.output_share) * a.in_price + a.output_share * a.out_price  # $/1M tokens
replica_hr = a.gpu_hr * a.gpus
dedicated_per_m = replica_hr / (a.tokens_per_gpu_hour / 1e6)  # $/1M tokens at 100% utilization

print(f"Serverless blended price   : ${blended:.3f} / 1M tokens")
print(f"Dedicated @100% utilization: ${dedicated_per_m:.3f} / 1M tokens (replica ${replica_hr:.2f}/hr)")
print(f"Break-even utilization     : {min(dedicated_per_m / blended, 9.99):.0%}\n")
print(f"{'tokens/month':>14} {'serverless':>12} {'dedicated(24/7)':>16}  cheaper")
for monthly in [1e8, 5e8, 1e9, 5e9, 1e10, 5e10]:
    s = monthly / 1e6 * blended
    replicas = max(1, -(-monthly // (a.tokens_per_gpu_hour * 730)))  # ceil
    d = replicas * replica_hr * 730
    print(f"{monthly:>14.0e} ${s:>11,.0f} ${d:>15,.0f}  {'dedicated' if d < s else 'serverless'}")
print("\nRemember: dedicated also buys latency isolation, custom models, and region control.")
