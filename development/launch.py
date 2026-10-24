"""Own one external-HMZ task deadline, author loop, confirmation and release."""
from __future__ import annotations
from hashlib import sha256
import json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path.cwd().resolve()

def write_new(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def read(path): return json.loads(path.read_text())

# scripts/install_hmz_maca.sh builds this copy of hmz 0.2.1: its stock fence grants
# /dev/nvidia*, /dev/dri and /dev/kfd but not MetaX's /dev/mxcd.
HMZ=Path.home()/'.cake-c500-hmz/bin/hmz'
MACA_ENVIRONMENT={'LD_LIBRARY_PATH':'/opt/maca/lib:/opt/maca/mxgpu_llvm/lib','MACA_PATH':'/opt/maca'}

def hmz_entry():
    if not os.access(HMZ,os.X_OK):raise RuntimeError('run-private hmz is unavailable: '+str(HMZ))
    subprocess.run([str(HMZ),'--help'],check=True,timeout=30,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return HMZ

def provider_environment(binding):
    """The C500 account's own Claude settings reach hmz through HOME; no key is copied."""
    if not ROOT.is_relative_to(binding['home']):
        raise RuntimeError('hmz fences writes only for a Run under the home directory: '+str(ROOT))
    env=dict(os.environ,HOME=binding['home']);entry=hmz_entry()
    env['PATH']=os.pathsep.join((str(entry.parent),env.get('PATH','')))
    # ~/.triton is read-only inside the fence, and a full evaluation outlasts Claude
    # Code's default two-minute Bash limit.
    env['TRITON_CACHE_DIR']=str(ROOT/'.local/triton-cache')
    env['BASH_DEFAULT_TIMEOUT_MS']=env['BASH_MAX_TIMEOUT_MS']=str(1800*1000)
    # glm-5.3 at high effort exceeded Claude Code's 32000-token reply cap in one
    # cohort-010 turn and the single-session author ended 73 minutes early.
    env['CLAUDE_CODE_MAX_OUTPUT_TOKENS']='64000'
    # The broker worker inherits this environment, and MACA torch cannot import without
    # its libraries on the path (OSError: libmxomp.so); run-007's launcher exported both.
    env.update(MACA_ENVIRONMENT)
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
        # The receipt names its own statistic; the C500 Compiler records medians only.
        statistic=timing.get('statistic','median')
        latency=(timing.get(f'pooled_{statistic}s_ms') or {}).get('candidate')
        if row.get('phase')=='search' and row.get('status')=='accepted' and timing.get('measurement_quality_passed') is True and isinstance(latency,(int,float)) and key in profiles:
            rows.append((latency,p.parent.name,row,profiles[key]))
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
    error=None;model_exit=None;confirmation=None;confirmation_timeout=None
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
                try:
                    code=execute(ROOT,{'candidate':candidate,'id':'final-confirmation',
                        'phase':'confirmation','route':route},env,seconds,ROOT/'campaign/logs')
                except subprocess.TimeoutExpired as exc:
                    confirmation_timeout=exc
                    confirmation={'status':'budget_exceeded','phase':'confirmation','error':str(exc)}
                else:
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
        if confirmation_timeout is not None:raise confirmation_timeout
        return 0
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        write_new(ROOT/'campaign/controller-status.json',{'model_exit':model_exit,'error':error,
            'done_exists':(ROOT/'campaign/DONE.json').exists(),'finished_at_epoch':time.time()})
if __name__=='__main__':raise SystemExit(main())
