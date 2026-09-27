"""Inspect and delete deployments over the REST control plane.

Usage:
  python 09_deployments/manage.py list
  python 09_deployments/manage.py get <deployment-id>
  python 09_deployments/manage.py delete <deployment-id>

Creating deployments is easiest with firectl (see lesson.md); this script shows the REST side.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from fwlearn.control import delete, get

cmd, arg = (sys.argv[1:] + [None, None])[:2]

if cmd == "list":
    deps = get("deployments", pageSize=200).get("deployments", [])
    print(f"{'ID':<30} {'STATE':<12} {'MODEL':<55} REPLICAS")
    for d in deps:
        print(f"{d['name'].split('/')[-1]:<30} {d.get('state', ''):<12} {d.get('baseModel', ''):<55} "
              f"{d.get('replicaCount', '?')} (min {d.get('minReplicaCount', 0)}, max {d.get('maxReplicaCount', '?')})")
    if not deps:
        print("(no deployments: nothing is costing you money)")
elif cmd == "get" and arg:
    print(json.dumps(get(f"deployments/{arg}"), indent=2))
elif cmd == "delete" and arg:
    if input(f"Delete deployment {arg}? [y/N] ").lower() == "y":
        delete(f"deployments/{arg}")
        print("deleted")
else:
    print(__doc__)
