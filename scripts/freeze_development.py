#!/usr/bin/env python3
"""Freeze the retained 54 C550 task identities into one external-HMZ plan."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / "development/tasks.json").read_text())["tasks"]
PCIS = ("0000:0f:00", "0000:34:00", "0000:48:00", "0000:5a:00",
        "0000:87:00", "0000:ae:00", "0000:c2:00", "0000:d7:00")


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError(f"expected object: {path}")
    return value


def host_rows(path, host):
    document = read(path)
    rows = document.get("rows")
    if (document.get("execution_host") != host or document.get("eligible_tasks") != 54
            or not isinstance(rows, list) or document.get("tasks") != len(rows)):
        raise ValueError(f"retained index differs for {host}")
    result=[]
    for row in rows:
        if row.get("execution_host") != host or row.get("task") not in TASKS:
            raise ValueError("retained row differs from its host/task denominator")
        required=("workload","starter","run","prepared_baseline")
        if any(not isinstance(row.get(name),str) or not Path(row[name]).is_absolute() for name in required):
            raise ValueError("retained task source paths differ")
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--c550-1-index',type=Path,required=True);p.add_argument('--c550-2-index',type=Path,required=True)
    p.add_argument('--compiler-commit',required=True);p.add_argument('--adapter-commit',required=True)
    for host in ('c550-1','c550-2'):
        key=host.replace('-','_')
        p.add_argument(f'--{host}-compiler-root',dest=key+'_compiler_root',required=True)
        p.add_argument(f'--{host}-adapter-root',dest=key+'_adapter_root',required=True)
        p.add_argument(f'--{host}-home',dest=key+'_home',required=True)
        p.add_argument(f'--{host}-owner-root',dest=key+'_owner_root',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(not re.fullmatch(r'[0-9a-f]{40}',x) for x in (a.compiler_commit,a.adapter_commit)):
        raise ValueError('commits must be full lowercase Git identities')
    rows=host_rows(a.c550_1_index,'c550-1')+host_rows(a.c550_2_index,'c550-2')
    if len(rows)!=54 or {r['task'] for r in rows}!=set(TASKS):
        raise ValueError('two retained indexes must cover every task exactly once')
    hosts={}
    assignments=[]
    for host in ('c550-1','c550-2'):
        key=host.replace('-','_')
        values={name:getattr(a,key+'_'+name) for name in ('compiler_root','adapter_root','home','owner_root')}
        if any(not Path(value).is_absolute() for value in values.values()):
            raise ValueError('host paths must be absolute')
        hosts[host]={**values,'devices':[{'physical_device':i,'runtime_device':0,'expected_pci':pci}
                                        for i,pci in enumerate(PCIS)]}
    for row in rows:
        host=row['execution_host']; physical=row['slot']['physical_device']
        assignments.append({'task':row['task'],'host':host,
            'root':str(Path(hosts[host]['owner_root'])/row['task']),
            'device':hosts[host]['devices'][physical],
            'source':{'workload':row['workload'],'starter':row['starter'],
                      'prior_run':row['run'],
                      'prepared_baseline':row['prepared_baseline']}})
    document={'schema_version':1,'kind':'c550_hmz_development_round','target':'xcore1002',
        'compiler_commit':a.compiler_commit,'adapter_commit':a.adapter_commit,
        'controls':{'model':'glm-5.3','reasoning_effort':'high','wall_time_seconds':10800,
            'search_seconds':9720,'confirmation_seconds':1080,'tokens':'accounting_only',
            'authors_per_device':2,'reference_access':'known_kernel_reproduction'},
        'skills':{'evolve':'skills/evolve/SKILL.md',
                  'metax-c550-hardware-exploration':'skills/metax-c550-hardware-exploration/SKILL.md'},
        'hosts':hosts,'assignments':assignments,
        'unsupported':[{'task':'tinygemm','status':'unsupported_target','target':'xcore1002'}],
        'stopping_rule':'all 54 assignments classified; no replacement Runs or budget reset'}
    with a.output.open('x') as stream: json.dump(document,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'tasks':54,'hosts':{h:sum(r['host']==h for r in assignments) for h in hosts},
                      'output':str(a.output.resolve())},indent=2))
if __name__=='__main__':main()
