"""Own one external-HMZ task deadline, author loop, confirmation and release."""
from __future__ import annotations
from hashlib import sha256
import json,os,signal,subprocess,sys,time
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
    from development.submission import abandon_pending,service
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    stop=time.monotonic()+timeout
    try:
        while p.poll() is None:
            if time.monotonic()>=stop:raise subprocess.TimeoutExpired(command,timeout)
            service(ROOT,env)
            time.sleep(0.2)
        return p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        return 124
    finally:
        abandon_pending(ROOT,'author process exited or deadline closed before response')

def hmz_command(entry,task,seconds):
    return [str(entry),'exec','-f','./development/ralph_flow.py',
            '-a','agent=claude/glm-5.3:high',
            '-p',f'budget.duration={seconds},budget.graceful=false',task]

def select_nominee():
    profiles={}
    for p in (ROOT/'campaign/evaluations').glob('*/outcome.json'):
        row=read(p)
        key=(row.get('candidate_path'),row.get('route'),row.get('source_sha256'),row.get('candidate_sha256'))
        if (row.get('phase')=='attribution' and row.get('status')=='accepted'
                and all(isinstance(value,str) for value in key)):
            profiles[key]=p.parent.name
    rows=[]
    for p in (ROOT/'campaign/evaluations').glob('*/outcome.json'):
        row=read(p);timing=row.get('timing') or {}
        key=(row.get('candidate_path'),row.get('route'),row.get('source_sha256'),row.get('candidate_sha256'))
        mean=(timing.get('pooled_means_ms') or {}).get('candidate')
        if row.get('phase')=='search' and row.get('status')=='accepted' and timing.get('measurement_quality_passed') is True and isinstance(mean,(int,float)) and key in profiles:
            rows.append((mean,p.parent.name,row,profiles[key]))
    return min(rows,key=lambda x:x[0]) if rows else None

def mentions_root(command,needle):
    start=0
    while True:
        index=command.find(needle,start)
        if index<0:return False
        after=index+len(needle)
        if after==len(command) or command[after] in b'/\0\" ,}]:':return True
        start=after

def live_task_processes(root=ROOT,proc_root=Path('/proc')):
    own=os.getpid();found=[];needle=str(root).encode()
    if not proc_root.is_dir():raise RuntimeError('process custody is unavailable')
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name)==own:continue
        try:
            cwd=os.readlink(proc/'cwd').encode();command=(proc/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError,ProcessLookupError,OSError):continue
        if cwd==needle or cwd.startswith(needle+b'/') or mentions_root(command,needle):found.append(int(proc.name))
    return sorted(found)

def release_verified(root=ROOT,proc_root=Path('/proc')):
    return not live_task_processes(root,proc_root)

def main(argv=None):
    from development.controller_entry import controller_command
    arguments=sys.argv[1:] if argv is None else argv
    check=arguments==['--check']
    if arguments not in ([],['--check']):raise ValueError('launch accepts only --check')
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
            subprocess.run(controller_command(ROOT,'prepare'),env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,
                           timeout=max(1,deadline['search_stop_at_epoch']-time.time()))
        remaining=max(1,int(deadline['search_stop_at_epoch']-time.time()))
        task=(ROOT/'campaign/TASK.md').read_text()
        with (ROOT/'campaign/logs/ralph.log').open('x') as log:
            model_exit=run_process(hmz_command(entry,task,remaining),env,remaining,log)
        nominee=select_nominee()
        if nominee:
            _,search_id,row,profile_id=nominee;candidate=row['candidate_path'];route=row['route']
            current=sha256((ROOT/candidate).read_bytes()).hexdigest()
            if current!=row['source_sha256']:
                confirmation={'status':'candidate_changed_before_confirmation',
                    'search_id':search_id,'profile_id':profile_id,
                    'expected_source_sha256':row['source_sha256'],'observed_source_sha256':current}
            else:
                seconds=max(1,int(deadline['stop_at_epoch']-time.time()))
                from development.submission import execute
                code=execute(ROOT,{'candidate':candidate,'id':'final-confirmation',
                    'phase':'confirmation','route':route},env,seconds,ROOT/'campaign/logs')
                path=ROOT/'campaign/evaluations/final-confirmation/outcome.json'
                confirmation=read(path) if path.exists() else {'status':'missing','exit_code':code}
                if (confirmation.get('source_sha256')!=row['source_sha256']
                        or confirmation.get('candidate_sha256')!=row['candidate_sha256']):
                    confirmation={'status':'candidate_identity_mismatch',
                        'search_id':search_id,'profile_id':profile_id,'observed':confirmation}
        live=live_task_processes();released=not live
        status='completed' if confirmation and confirmation.get('status')=='accepted' and released else 'no_qualified_result'
        write_new(ROOT/'campaign/DONE.json',{'terminal':True,'status':status,'model_exit':model_exit,
            'device_release_verified':released,'release_basis':'no_live_task_processes',
            'live_task_processes':live,'confirmation':confirmation})
        return 0
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        write_new(ROOT/'campaign/controller-status.json',{'model_exit':model_exit,'error':error,
            'done_exists':(ROOT/'campaign/DONE.json').exists(),'finished_at_epoch':time.time()})
if __name__=='__main__':raise SystemExit(main())
