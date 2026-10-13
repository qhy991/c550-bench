"""Own one external-HMZ task deadline, author loop, confirmation and release."""
from __future__ import annotations
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path.cwd().resolve()

def write_new(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def read(path): return json.loads(path.read_text())

def hmz_entry():
    entry=Path(sys.executable).absolute().parent/'hmz'
    if not entry.is_file() or not os.access(entry,os.X_OK):raise RuntimeError('hmz is unavailable beside owner Python')
    subprocess.run([str(entry),'--help'],check=True,timeout=30,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return entry

def provider_environment(binding):
    env=dict(os.environ,HOME=binding['home']);entry=hmz_entry();bin_dir=str(entry.parent)
    env['PATH']=os.pathsep.join((bin_dir,'/provider-native',env.get('PATH','')))
    credential=Path('/provider-auth/credential').read_text().strip()
    if not credential:raise RuntimeError('provider credential is empty')
    env.update(ANTHROPIC_AUTH_TOKEN=credential,ANTHROPIC_API_KEY=credential,
               ANTHROPIC_BASE_URL='https://cloud.infini-ai.com/maas',ANTHROPIC_MODEL='glm-5.3',
               ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.3',ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.3',
               ANTHROPIC_DEFAULT_HAIKU_MODEL='glm-5.3')
    return entry,env

def run_process(command,env,timeout,log):
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        return 124

def hmz_command(entry,task):
    return [str(entry),'exec','-f','development/ralph_flow.py',
            '-a','claude/glm-5.3:high',task]

def select_nominee():
    profiles={}
    for p in (ROOT/'campaign/evaluations').glob('*/outcome.json'):
        row=read(p)
        if row.get('phase')=='attribution' and row.get('status')=='accepted': profiles[(row.get('candidate_path'),row.get('route'))]=p.parent.name
    rows=[]
    for p in (ROOT/'campaign/evaluations').glob('*/outcome.json'):
        row=read(p);timing=row.get('timing') or {};key=(row.get('candidate_path'),row.get('route'))
        mean=(timing.get('pooled_means_ms') or {}).get('candidate')
        if row.get('phase')=='search' and row.get('status')=='accepted' and timing.get('measurement_quality_passed') is True and isinstance(mean,(int,float)) and key in profiles:
            rows.append((mean,p.parent.name,row,profiles[key]))
    return min(rows,key=lambda x:x[0]) if rows else None

def release_verified(binding):
    lock=Path(f"/tmp/open-cake-ir-maca-0-device-{binding['device']['physical_device']}.lock")
    with lock.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        finally:
            try:fcntl.flock(f,fcntl.LOCK_UN)
            except OSError:pass
    return True

def main(argv=None):
    check=(argv or sys.argv[1:])==['--check']
    if (argv or sys.argv[1:]) not in ([],['--check']):raise ValueError('launch accepts only --check')
    binding=read(ROOT/'campaign/development-binding.json')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('prepared task source changed')
    for relative in binding['skills'].values():
        if not (ROOT/'.deps/cake-ir'/relative).is_file():raise ValueError('required skill is unavailable: '+relative)
    entry,env=provider_environment(binding)
    subprocess.run([sys.executable,'campaign/compiler_tools.py','catalog'],cwd=ROOT,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if check:print('binding, skills, frozen HMZ catalog and author entry are available');return 0
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    now=time.time();deadline={'source_commit':source_commit,'started_at_epoch':now,'search_stop_at_epoch':now+binding['search_seconds'],
        'stop_at_epoch':now+binding['wall_time_seconds'],'wall_time_seconds':binding['wall_time_seconds'],
        'confirmation_seconds':binding['confirmation_seconds'],'preparation_cost_included':True}
    write_new(ROOT/'campaign/deadline.json',deadline)
    write_new(ROOT/'campaign/intake.json',{'source_commit':source_commit,
        'task':binding['task'],'compiler':binding['compiler'],'model':'claude/glm-5.3:high',
        'skills':binding['skills'],'routes':binding['routes']})
    error=None;model_exit=None;confirmation=None
    try:
        with (ROOT/'campaign/logs/preparation.log').open('x') as log:
            subprocess.run([sys.executable,'-m','development.runtime','prepare'],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,
                           timeout=max(1,deadline['search_stop_at_epoch']-time.time()))
        remaining=max(1,int(deadline['search_stop_at_epoch']-time.time()))
        task=(ROOT/'campaign/TASK.md').read_text()
        with (ROOT/'campaign/logs/ralph.log').open('x') as log:
            model_exit=run_process(hmz_command(entry,task),env,remaining,log)
        nominee=select_nominee()
        if nominee:
            _,search_id,row,profile_id=nominee;candidate=row['candidate_path'];route=row['route']
            seconds=max(1,int(deadline['stop_at_epoch']-time.time()))
            with (ROOT/'campaign/logs/confirmation.log').open('x') as log:
                code=subprocess.call(['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',
                    sys.executable,'-m','development.runtime','evaluate','--candidate',candidate,
                    '--id','final-confirmation','--phase','confirmation','--route',route],env=env,stdout=log,stderr=subprocess.STDOUT)
            path=ROOT/'campaign/evaluations/final-confirmation/outcome.json'
            confirmation=read(path) if path.exists() else {'status':'missing','exit_code':code}
        released=release_verified(binding)
        status='completed' if confirmation and confirmation.get('status')=='accepted' and released else 'no_qualified_result'
        write_new(ROOT/'campaign/DONE.json',{'terminal':True,'status':status,'model_exit':model_exit,
            'device_release_verified':released,'confirmation':confirmation})
        return 0
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        write_new(ROOT/'campaign/controller-status.json',{'model_exit':model_exit,'error':error,
            'done_exists':(ROOT/'campaign/DONE.json').exists(),'finished_at_epoch':time.time()})
if __name__=='__main__':raise SystemExit(main())
