import argparse, json
from pathlib import Path
from .prepare import prepare
p=argparse.ArgumentParser()
p.add_argument('--plan',type=Path,required=True);p.add_argument('--host',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
print(json.dumps(prepare(a.plan,a.host,a.output),indent=2))
