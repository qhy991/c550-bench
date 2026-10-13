"""Select never-claimed tasks without resetting any historical author opportunity."""
import argparse
import json
from pathlib import Path

from development.binding import git, read, validate
from development.prepare import prepare
from development.queue import process_alive


def unclaimed(plan_path, host, previous_allocation, previous_state, repository):
    plan=read(Path(plan_path));rows=validate(plan,Path(repository));old=read(Path(previous_allocation))
    state=Path(previous_state);launch=read(state/'LAUNCH.json')
    if not (state/'claims').is_dir():
        raise ValueError('predecessor claim custody is unavailable')
    if (old.get('kind')!='c550_hmz_development_allocation' or old.get('host')!=host
            or old.get('compiler')!=plan['compiler_commit']):
        raise ValueError('predecessor allocation differs from the host and Compiler')
    if Path(launch['allocation']).resolve()!=Path(previous_allocation).resolve():
        raise ValueError('predecessor queue names a different allocation')
    if process_alive({'pid':launch['owner_pid'],'start_ticks':launch.get('owner_start_ticks')}):
        raise ValueError('predecessor queue can still claim tasks')
    current={row['task']:row for row in rows if row['host']==host}
    previous={run['task']:run for run in old['runs']}
    if len(previous)!=len(old['runs']) or set(previous)!=set(current):
        raise ValueError('predecessor host denominator differs')
    for task,run in previous.items():
        if not Path(run['root']).is_dir():
            raise ValueError('predecessor workspace is unavailable')
        if run['device']!=current[task]['device']:
            raise ValueError('predecessor device route differs')
        if run['source_provenance']!=current[task]['source']:
            raise ValueError('retained task material differs')
    claimed=set()
    for path in (state/'claims').glob('*.json'):
        value=read(path);task=value.get('task')
        if (task not in previous or path.stem!=task
                or value.get('author_opportunity_consumed') is not True):
            raise ValueError('unreadable or foreign historical claim')
        claimed.add(task)
    for directory in ('commands','owners'):
        for path in (state/directory).glob('*.json'):
            if path.stem not in previous:raise ValueError('foreign predecessor execution record')
            claimed.add(path.stem)
    for path in (state/'logs').glob('*'):
        if path.suffix in {'.stdout','.stderr'}:
            if path.stem not in previous:raise ValueError('foreign predecessor log')
            claimed.add(path.stem)
    for path in (state/'terminals').glob('*.json'):
        value=read(path)
        if path.stem not in previous:raise ValueError('foreign predecessor terminal')
        if not (value.get('error') in {'unstarted_owner_exit','unstarted_release_blocked'}
                and value.get('done') is None and value.get('exit_code') is None):
            claimed.add(path.stem)
    for task,run in previous.items():
        root=Path(run['root'])/'campaign'
        if (root/'intake.json').exists() or (root/'deadline.json').exists() or (root/'DONE.json').exists():
            claimed.add(task)
    for task in set(previous)-claimed:
        run=previous[task];root=Path(run['root'])
        if git(root,'rev-parse','HEAD')!=run['prepared_commit']:
            raise ValueError('predecessor unclaimed workspace identity differs')
        binding=read(root/'campaign/development-binding.json')
        if binding!={k:v for k,v in run.items() if k!='prepared_commit'}:
            raise ValueError('predecessor unclaimed binding differs')
    return frozenset(set(previous)-claimed),frozenset(claimed)


def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','previous-allocation','previous-state','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--host',required=True);args=parser.parse_args()
    repository=Path(__file__).resolve().parents[1]
    selected,excluded=unclaimed(args.plan,args.host,args.previous_allocation,args.previous_state,repository)
    allocation=prepare(args.plan,args.host,args.output,repository=repository,tasks=selected)
    print(json.dumps({'prepared':len(allocation['runs']),'excluded_historical':sorted(excluded),
                      'tasks':sorted(selected),'provider_calls':0,'gpu_calls':0},indent=2))


if __name__=='__main__':main()
