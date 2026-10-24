"""Finite at-most-once HMZ task sequencing for one prepared C500 host."""
from __future__ import annotations
from collections import Counter
import argparse,fcntl,json,os,shlex,subprocess,time
from pathlib import Path


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def read(path):return json.loads(Path(path).read_text())
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def unstarted_terminal(task,value):
    keys={'task','exit_code','done','release_verified','error'}
    if value.get('error')=='unstarted_owner_exit':keys.add('reconciled_after_owner_exit')
    return (set(value)==keys and value.get('task')==task
        and value.get('error') in {'unstarted_owner_exit','unstarted_release_blocked'}
        and value.get('done') is None and value.get('exit_code') is None
        and value.get('release_verified') is False
        and (value.get('error')!='unstarted_owner_exit'
             or value.get('reconciled_after_owner_exit') is True))
def process_start_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except (FileNotFoundError,ProcessLookupError,PermissionError,OSError):return None
def process_alive(identity):
    pid=identity.get('pid');ticks=identity.get('start_ticks')
    if not isinstance(pid,int):return False
    current=process_start_ticks(pid)
    if isinstance(ticks,str) and current is not None:return current==ticks
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


class ContainerLauncher:
    def __init__(self,namespace,container,python):
        self.namespace=namespace;self.container=container;self.python=python
    def start(self,run,claim,state):
        entry=(Path(run['root'])/'.deps/launcher/development/controller_entry.py').resolve(strict=True)
        root=Path(run['root']);command=['/usr/local/bin/nerdctl','-n',self.namespace,'exec',
            '--workdir',str(root),self.container,self.python,'-I',str(entry),'--root',str(root),'launch']
        write_new(state/'commands'/(run['task']+'.json'),command)
        out=(state/'logs'/(run['task']+'.stdout')).open('x');err=(state/'logs'/(run['task']+'.stderr')).open('x')
        process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        write_new(state/'owners'/(run['task']+'.json'),{'pid':process.pid,
            'start_ticks':process_start_ticks(process.pid)})
        return process
    def terminal(self,run,process,code):
        done=Path(run['root'])/'campaign/DONE.json'
        document=read(done) if done.is_file() else None
        return {'task':run['task'],'exit_code':code,'done':document,
                'release_verified':bool(document and document.get('device_release_verified') is True)}


class HostLauncher:
    """C500 runs each task's controller on the host under the MetaX render group.

    The author is fenced by hmz, and device work goes through the existing MACA broker,
    so there is no per-task container. controller_entry refuses a writable launcher
    checkout; bubblewrap binds that checkout read-only over itself, as the C550
    container mount did, and leaves the rest of the host as it is.
    """
    def __init__(self,python):
        self.python=python
    def start(self,run,claim,state):
        launcher=(Path(run['root'])/'.deps/launcher').resolve(strict=True)
        entry=launcher/'development/controller_entry.py'
        root=Path(run['root'])
        inner=['/usr/bin/bwrap','--dev-bind','/','/','--ro-bind',str(launcher),str(launcher),
               '--chdir',str(root),self.python,'-I',str(entry),'--root',str(root),'launch']
        command=['/usr/bin/sg','render','-c',' '.join(shlex.quote(part) for part in inner)]
        write_new(state/'commands'/(run['task']+'.json'),command)
        out=(state/'logs'/(run['task']+'.stdout')).open('x');err=(state/'logs'/(run['task']+'.stderr')).open('x')
        process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        write_new(state/'owners'/(run['task']+'.json'),{'pid':process.pid,
            'start_ticks':process_start_ticks(process.pid)})
        return process
    terminal=ContainerLauncher.terminal


class DevelopmentQueue:
    def __init__(self,allocation,state,launcher,sleeper=time.sleep):
        self.allocation=Path(allocation).resolve(strict=True);self.state=Path(state)
        self.launcher=launcher;self.sleeper=sleeper;self.document=read(self.allocation)
        if self.document.get('kind')!='c500_hmz_development_allocation':raise ValueError('unsupported allocation')
        # Authors per device comes from the frozen binding: C550 ran two, C500 runs one.
        capacities={run.get('authors_per_device') for run in self.document['runs']}
        if len(capacities)!=1 or not isinstance(next(iter(capacities)),int) or next(iter(capacities))<1:
            raise ValueError('allocation must declare one positive authors_per_device')
        self.capacity=capacities.pop()
        tasks=set()
        for run in self.document.get('runs',[]):
            root=Path(run.get('root',''))
            if run.get('task') in tasks or not root.is_dir() or git(root,'rev-parse','HEAD')!=run.get('prepared_commit'):
                raise ValueError('prepared development allocation differs')
            binding=read(root/'campaign/development-binding.json')
            if binding!={k:v for k,v in run.items() if k!='prepared_commit'}:raise ValueError('prepared binding differs')
            if (root/'campaign/deadline.json').exists() or (root/'campaign/DONE.json').exists():
                raise ValueError('allocation contains an attempted task')
            tasks.add(run['task'])
    def run(self):
        self.state.mkdir(parents=True,exist_ok=False)
        for d in ('claims','commands','owners','logs','terminals'): (self.state/d).mkdir()
        with (self.state/'owner.lock').open('x') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write_new(self.state/'LAUNCH.json',{'owner_pid':os.getpid(),'started_at_epoch':time.time(),
                'owner_start_ticks':process_start_ticks(os.getpid()),
                'allocation':str(self.allocation),'tasks':len(self.document['runs']),
                'capacity_per_device':self.capacity,'automatic_retries':0})
            pending=list(self.document['runs']);active={};blocked=set()
            while pending or active:
                for task,(run,handle) in list(active.items()):
                    code=handle.poll()
                    if code is None:continue
                    try:terminal=self.launcher.terminal(run,handle,code)
                    except Exception as e:terminal={'task':task,'exit_code':code,'done':None,
                        'release_verified':False,'error':type(e).__name__}
                    write_new(self.state/'terminals'/(task+'.json'),terminal)
                    if not terminal['release_verified']:blocked.add(run['device']['physical_device'])
                    del active[task]
                counts=Counter(run['device']['physical_device'] for run,_ in active.values())
                while True:
                    index=next((i for i,r in enumerate(pending)
                        if r['device']['physical_device'] not in blocked and counts[r['device']['physical_device']]<self.capacity),None)
                    if index is None:break
                    run=pending.pop(index);task=run['task'];claim={'task':task,
                        'physical_device':run['device']['physical_device'],'claimed_at_epoch':time.time(),
                        'author_opportunity_consumed':True}
                    write_new(self.state/'claims'/(task+'.json'),claim)
                    try:handle=self.launcher.start(run,claim,self.state)
                    except Exception as e:
                        write_new(self.state/'terminals'/(task+'.json'),{'task':task,'exit_code':None,
                            'done':None,'release_verified':False,'error':'launch:'+type(e).__name__})
                        blocked.add(run['device']['physical_device']);continue
                    active[task]=(run,handle);counts[run['device']['physical_device']]+=1
                if not active and pending:
                    for run in pending:write_new(self.state/'terminals'/(run['task']+'.json'),{
                        'task':run['task'],'exit_code':None,'done':None,'release_verified':False,
                        'error':'unstarted_release_blocked'})
                    pending.clear();break
                self.sleeper(1)
            summary={'assigned':len(self.document['runs']),'started':len(list((self.state/'claims').glob('*.json'))),
                'terminal':len(list((self.state/'terminals').glob('*.json'))),'blocked_devices':sorted(blocked)}
            write_new(self.state/'DONE.json',summary);return summary


def reconcile(allocation,state):
    """Seal every assignment after a dead owner, without starting or retrying work."""
    allocation=Path(allocation).resolve(strict=True);state=Path(state).resolve(strict=True)
    document=read(allocation);launch=read(state/'LAUNCH.json')
    if document.get('kind')!='c500_hmz_development_allocation':raise ValueError('unsupported allocation')
    if (state/'DONE.json').exists():raise FileExistsError('queue is already terminal')
    owner={'pid':launch.get('owner_pid'),'start_ticks':launch.get('owner_start_ticks')}
    if process_alive(owner):raise ValueError('queue owner is still live')
    for path in (state/'owners').glob('*.json'):
        if process_alive(read(path)):raise ValueError('claimed task process is still live')
    blocked=set()
    for run in document['runs']:
        task=run['task'];terminal=state/'terminals'/(task+'.json')
        if terminal.exists():continue
        claimed=(state/'claims'/(task+'.json')).exists();done=Path(run['root'])/'campaign/DONE.json'
        if claimed and done.is_file():
            result=read(done);released=result.get('device_release_verified') is True
            row={'task':task,'exit_code':None,'done':result,'release_verified':released,
                 'reconciled_after_owner_exit':True}
            if not released:blocked.add(run['device']['physical_device'])
        else:
            row={'task':task,'exit_code':None,'done':None,'release_verified':False,
                 'error':'unknown_owner_exit' if claimed else 'unstarted_owner_exit',
                 'reconciled_after_owner_exit':True}
            if claimed:blocked.add(run['device']['physical_device'])
        write_new(terminal,row)
    summary={'assigned':len(document['runs']),
        'started':len(list((state/'claims').glob('*.json'))),
        'terminal':len(list((state/'terminals').glob('*.json'))),
        'blocked_devices':sorted(blocked),'reconciled_after_owner_exit':True}
    write_new(state/'DONE.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--namespace');p.add_argument('--container');p.add_argument('--python');p.add_argument('--reconcile',action='store_true')
    p.add_argument('--host',action='store_true',help='run each controller on this host (C500)')
    a=p.parse_args()
    if a.reconcile:result=reconcile(a.allocation,a.state)
    elif a.host:
        if not a.python:p.error('--host requires --python')
        result=DevelopmentQueue(a.allocation,a.state,HostLauncher(a.python)).run()
    else:
        if not all((a.namespace,a.container,a.python)):p.error('launch requires --namespace, --container and --python')
        result=DevelopmentQueue(a.allocation,a.state,
            ContainerLauncher(a.namespace,a.container,a.python)).run()
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
