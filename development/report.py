"""Derive development status from retained allocations and evaluation evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from campaign.report import _read
from development.queue import process_alive


def observation(path):
    row,error=_read(path)
    return row, {'path':str(path),'read_error':error} if error else {'path':str(path)}


def report(allocation_path,state):
    allocation=json.loads(Path(allocation_path).read_text());state=Path(state)
    if allocation.get('kind')!='c550_hmz_development_allocation':
        raise ValueError('unsupported development allocation')
    rows=[];seen=set()
    for run in allocation['runs']:
        task=run['task'];root=Path(run['root'])
        if task in seen:raise ValueError('duplicate task in allocation')
        seen.add(task)
        claim,claim_ref=observation(state/'claims'/f'{task}.json')
        terminal,terminal_ref=observation(state/'terminals'/f'{task}.json')
        owner,owner_ref=observation(state/'owners'/f'{task}.json')
        done,done_ref=observation(root/'campaign/DONE.json')
        deadline,deadline_ref=observation(root/'campaign/deadline.json')
        evaluations=[]
        for path in sorted((root/'campaign/evaluations').glob('*/outcome.json')):
            value,ref=observation(path)
            if value is None:
                evaluations.append({'id':path.parent.name,'status':'unknown_unreadable','evidence':ref});continue
            evaluations.append({'id':path.parent.name,'phase':value.get('phase'),
                'route':value.get('route'),'candidate_path':value.get('candidate_path'),
                'status':value.get('status'),'correctness_passed':value.get('correctness_passed'),
                'timing':value.get('timing'),'evidence':ref})
        if terminal is not None:execution='terminal'
        elif owner is not None and process_alive(owner):execution='running'
        elif done is not None:execution='task_terminal_owner_uncollected'
        elif claim is not None:execution='unknown_claimed_without_live_owner'
        else:execution='unstarted'
        confirmation=(done or {}).get('confirmation') or {}
        if not isinstance(confirmation,dict):confirmation={}
        timing=confirmation.get('timing') or {}
        if not isinstance(timing,dict):timing={}
        counts=timing.get('pooled_sample_counts') or {}
        qualified=(confirmation.get('status')=='accepted'
            and confirmation.get('correctness_passed') is True
            and timing.get('measurement_quality_passed') is True
            and counts=={'baseline':10,'candidate':10}
            and (done or {}).get('device_release_verified') is True)
        budget_met=None
        if deadline is not None and isinstance(confirmation.get('completed_at_epoch'),(float,int)):
            budget_met=confirmation['completed_at_epoch']<=deadline['stop_at_epoch']
        faults=[]
        for path in sorted((root/'campaign/evaluation-requests').glob('*/response.json')):
            value,ref=observation(path)
            if value is None or 'error' in value:
                faults.append({'id':path.parent.name,'error':(value or {}).get('error','unreadable'),
                               'evidence':ref})
        rows.append({'task':task,'execution':execution,'terminal_status':(done or {}).get('status'),
            'controller_exit_code':(terminal or {}).get('exit_code'),
            'terminal_error':(terminal or {}).get('error'),
            'model_exit':(done or {}).get('model_exit'),
            'evaluations':evaluations,'request_faults':faults,
            'confirmation':{'status':confirmation.get('status','missing'),
                'route':confirmation.get('route'),'candidate_path':confirmation.get('candidate_path'),
                'correctness_passed':confirmation.get('correctness_passed'),
                'timing':timing or None,'qualified':qualified,'budget_met':budget_met},
            'release_verified':(done or {}).get('device_release_verified'),
            'promotion_disposition':'unreviewed',
            'evidence':{'claim':claim_ref,'terminal':terminal_ref,'owner':owner_ref,
                'done':done_ref,'deadline':deadline_ref,
                'journal':str(root/'campaign/JOURNAL.md'),'evolution':str(root/'campaign/EVOLUTION.md')}})
    return {'host':allocation['host'],'compiler':allocation['compiler'],
        'launcher':allocation['adapter_commit'],'assigned':len(rows),
        'execution_counts':dict(Counter(row['execution'] for row in rows)),
        'rows':rows,'scope':'retained evidence projection; no optimization success or Compiler effect inferred'}


def combine(reports):
    registered=json.loads((Path(__file__).with_name('tasks.json')).read_text())
    rows=[row for report in reports for row in report['rows']]
    tasks=[row['task'] for row in rows]
    if len(set(tasks))!=len(tasks):raise ValueError('duplicate task across host reports')
    if set(tasks)!=set(registered['tasks']):raise ValueError('host reports do not cover the54 task denominator')
    if len({(r['compiler'],r['launcher']) for r in reports})!=1:
        raise ValueError('host reports use different Compiler or launcher')
    return {'assigned':54,'complete_denominator':True,'hosts':reports,
            'unsupported':registered['unsupported'],'performance_claim':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allocation',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(report(args.allocation,args.state),indent=2,sort_keys=True))


if __name__=='__main__':main()
