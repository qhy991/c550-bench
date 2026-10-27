"""CPU preparation and bounded candidate Evaluation for one prepared C550 HMZ task."""
from __future__ import annotations

from dataclasses import asdict, fields
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def write_new(path: Path, value: object) -> None:
    from open_cake_ir.evaluation.core import _plain_json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(_plain_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def evaluation_purpose(phase: str) -> str:
    return 'confirmatory' if phase == 'confirmation' else phase


def parse_cake_submission(text, filename, parser, frontend_error):
    """Keep known candidate-construction errors inside author feedback."""
    if Path(filename).suffix == '.json':
        try:
            document=json.loads(text)
            if not isinstance(document,dict):
                raise ValueError('Cake JSON candidate must be a Schedule or Program object')
            json.dumps(document,allow_nan=False)
        except (ValueError,TypeError) as error:
            feedback={'stage':'submission','error':str(error),'code':'CAKE_JSON_INVALID'}
            if isinstance(error,json.JSONDecodeError):
                feedback['source_location']={'filename':filename,'line':error.lineno,'column':error.colno}
            return None,feedback
        return document,None
    try:
        skeleton=parser(text,filename=filename)
    except frontend_error as error:
        return None,{'stage':'assessment','error':str(error),'code':error.code,
                     'source_location':asdict(error.location)}
    return ({'python_program_source':text,'program_id':skeleton['program_id']}
            if 'program_id' in skeleton else {'python_source':text}),None


def build_timeout_feedback(error):
    """Feedback for a candidate whose isolated compile ran out of time, else None.

    The build jail runs before any device admission, so a compile that exceeds its
    timeout costs only that candidate (fib_gemm_n5120_k2048, r1b: a [256, 2048] tile).
    Every other harness fault, and any timeout during device evaluation, still closes
    the search, because there the device and release state are not known.
    """
    from open_cake_ir.lab.faults import RunProtocolFault
    from open_cake_ir.lab.process import SupervisedProcessTimeout
    if not (isinstance(error, RunProtocolFault) and error.protocol_adherence == 'harness_fault'
            and isinstance(error.__cause__, SupervisedProcessTimeout)):
        return None
    return {'stage':'build','code':'BUILD_TIMEOUT','error':str(error),
            'repair':'the candidate or its toolchain exceeded the isolated build timeout before '
                     'reaching the device; a smaller compiled tile or unroll extent compiles faster'}


def reject_build_timeout(output, error, *, route, phase, source_sha256, candidate_sha256, candidate_path):
    """Record a build timeout as a rejected outcome; None for any other fault."""
    feedback=build_timeout_feedback(error)
    if feedback is None:return None
    if getattr(error,'artifact_payloads',None):retain(output/'build-artifacts',error.artifact_payloads)
    write_new(output/'build.json',{'route':route,'disposition':'rejected','feedback':feedback})
    result={'status':'rejected','route':route,'phase':phase,'feedback':feedback,
            'source_sha256':source_sha256,'candidate_sha256':candidate_sha256,
            'candidate_path':candidate_path,'completed_at_epoch':time.time()}
    write_new(output/'outcome.json',result)
    return result


def retain(root: Path, payloads) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for role, payload in payloads.items():
        if not role.isidentifier():
            raise ValueError('artifact role is not a file name')
        (root / role).write_bytes(payload)


def load_local_baseline(root: Path, bundle: Path):
    """Read the task's own generated baseline without weakening external custody."""
    base=(root/'campaign/baseline-preparation').resolve(strict=True)
    path=Path(bundle).resolve(strict=True)
    if base not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError('local baseline bundle escapes its preparation directory')
    from open_cake_ir.evaluation.paired import candidate_from_identity
    document=json.loads(path.read_text())
    identity=document.get('candidate')
    paths=document.get('artifact_paths')
    if not isinstance(identity,dict) or not isinstance(paths,dict):
        raise ValueError('local baseline bundle fields differ')
    payloads={}
    for role,name in paths.items():
        if (not isinstance(role,str) or not role.isidentifier() or not isinstance(name,str)
                or Path(name).is_absolute() or '..' in Path(name).parts):
            raise ValueError('local baseline artifact locator differs')
        artifact=(path.parent/name).resolve(strict=True)
        if path.parent!=artifact.parent or artifact.is_symlink() or not artifact.is_file():
            raise ValueError('local baseline artifact escapes its bundle')
        payloads[role]=artifact.read_bytes()
    return candidate_from_identity(identity,payloads)


def context(root: Path):
    binding=json.loads((root/'campaign/development-binding.json').read_text())
    source=(root/'.deps/cake-ir').resolve(strict=True)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=binding['compiler']:
        raise ValueError('pinned Compiler differs')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('pinned Compiler is not clean')
    sys.path[:0]=[str(source),str(source/'src')]
    return binding,source


def prepare(root: Path) -> dict:
    binding,source=context(root)
    from open_cake_ir.compiler import frontend
    from open_cake_ir.tasks.workloads import load_workload
    from tools.launch_task import _admit_stack,_prepare_baseline,_runtime_config
    workload_path=root/'campaign/original-workload.json';starter=root/'campaign/candidates/starter.py'
    workload=load_workload(workload_path)
    if workload.target!='xcore1000': raise ValueError('development Workload target differs')
    prior=json.loads((root/'campaign/prior-run.json').read_text())
    protocol=prior['evaluation_protocol'];authoring=json.loads(json.dumps(prior['authoring']))
    preparation=root/'campaign/baseline-preparation';preparation.mkdir(exist_ok=False)
    compiler,executor,host,compiler_ref=_admit_stack(source,preparation,
                                                      workload.target,'triton')
    authoring['compiler_revision']=compiler_ref
    bundle=_prepare_baseline(source,preparation,compiler,executor,host,
                             workload,authoring,starter.read_text(),compiler_ref,'triton')
    schedule=frontend.read_schedule(starter).document
    requirements=compiler.lower(compiler.assess(schedule)).toolchain_requirements
    write_new(root/'campaign/native-interface.json',{
        'schema_version':1,'route':'native_triton','target':workload.target,
        'kernel_entry_point':requirements['kernel_entry_point'],
        'compile_constants':dict(requirements['compile_constants']),
        'compile_options':dict(requirements['compile_options']),
        'grid':list(requirements['grid']),
        'source_contract':'one kernel definition with fixed Triton imports',
    })
    baseline=load_local_baseline(root,bundle)
    from open_cake_ir.evaluation.paired import candidate_identity
    prepared={'schema_version':1,'fixed_baseline_bundle_path':str(bundle),
              'fixed_baseline_candidate':candidate_identity(baseline),
              'fixed_baseline_selection':{'schema_version':1,'policy':'starter_reference',
                'source':'starter_reference','incumbent_key':None,'promotion_run_id':None,'registry_root':None}}
    write_new(root/'campaign/prepared-baseline.json',prepared)
    device=binding['device']
    runtime_root=root/'campaign/runtime';runtime_root.mkdir(exist_ok=False)
    runtime=_runtime_config(runtime_root,executor,Path('/bin/true'),'triton',
        allocation='local_broker',local_kind='maca',local_device=device['physical_device'],
        local_runtime_device=device['runtime_device'],local_expected_pci=device['expected_pci'],
        local_lock_scope='device',local_queue_seconds=300,cpu_threads=4)
    write_new(root/'campaign/runtime.json',runtime)
    write_new(root/'campaign/authority.json',{'authoring':authoring,'evaluation_protocol':protocol,
        'compiler_reference':compiler_ref,'workload_sha256':workload.canonical_sha256,
        'case_id':'primary','routes':['cake','native_triton_engineering']})
    return {'task':binding['task'],'baseline':str(bundle),'provider_calls':0,'gpu_calls':0}


def phase_broker(broker: dict, phase: str, stop: float, now: float) -> dict:
    """The final confirmation is attempted once while sibling authors on the card queue for its
    lease. It waits for the lease as long as its phase lasts instead of the search wait."""
    if phase != 'confirmation':
        return broker
    command = list(broker['command'])
    index = command.index('--local-queue-seconds') + 1
    search_wait = float(command[index])
    wait = max(search_wait, float(int(stop - now)))
    command[index] = str(int(wait))
    return dict(broker, command=command, timeout_seconds=broker['timeout_seconds'] + int(wait - search_wait))


class DeadlineAdmission:
    def __init__(self, root: Path, phase: str):
        deadline=json.loads((root/'campaign/deadline.json').read_text())
        self.stop=deadline['stop_at_epoch'] if phase=='confirmation' else deadline['search_stop_at_epoch']
    def before_submission(self, attempt: int) -> bool:
        return type(attempt) is int and 0<attempt<=2 and time.time()<self.stop


def native_plan(root: Path, candidate: Path, binding: dict) -> dict:
    folder=candidate.resolve().parent
    native=(root/'campaign/native-engineering').resolve()
    if native not in folder.parents:
        raise ValueError('native candidate must live under campaign/native-engineering/<id>')
    plan=json.loads((folder/'plan.json').read_text())
    expected={'schema_version':1,'route':'native_triton','task':binding['task'],
              'target':'xcore1000','cake_candidate':plan.get('cake_candidate'),
              'missing_commitment':plan.get('missing_commitment'),
              'same_contract':True,'shares_task_deadline':True}
    if plan!=expected or not isinstance(plan['cake_candidate'],str) or not plan['cake_candidate'] \
            or not isinstance(plan['missing_commitment'],str) or not plan['missing_commitment']:
        raise ValueError('native engineering plan differs from the bounded route split')
    cake=(root/plan['cake_candidate']).resolve()
    if (root/'campaign/candidates').resolve() not in cake.parents or not cake.is_file():
        raise ValueError('native plan must cite one retained own Cake candidate')
    return plan


def evaluate(root: Path, candidate: Path, identity: str, phase: str, route: str) -> dict:
    binding,source=context(root)
    if phase not in {'search','confirmation','attribution'}: raise ValueError('unknown evaluation phase')
    if route not in {'cake','native_triton'}: raise ValueError('unknown authoring route')
    deadline=json.loads((root/'campaign/deadline.json').read_text())
    cutoff=deadline['stop_at_epoch'] if phase=='confirmation' else deadline['search_stop_at_epoch']
    if time.time()>=cutoff: raise TimeoutError('task phase budget is closed')
    if not identity or not identity.replace('-','').replace('_','').isalnum(): raise ValueError('invalid evaluation id')
    output=root/'campaign/evaluations'/identity
    output.mkdir(parents=True,exist_ok=False)
    candidate_path=candidate.resolve(strict=True)
    try:
        candidate_relative=str(candidate_path.relative_to(root))
    except ValueError as error:
        raise ValueError('candidate must belong to this task workspace') from error
    candidate_bytes=candidate_path.read_bytes()
    source_sha256=sha256(candidate_bytes).hexdigest()
    suffix='py' if route=='cake' and candidate_path.suffix!='.json' else 'json'
    (output/('authored-source.'+suffix)).write_bytes(candidate_bytes)
    authority=json.loads((root/'campaign/authority.json').read_text())
    runtime=json.loads((root/'campaign/runtime.json').read_text())
    from open_cake_ir.compiler import frontend
    from open_cake_ir.lab.environments import CandidateSubmission,NativeTritonEnvironment
    from open_cake_ir.lab.runtime import BoundedBrokerEvaluator,CommandBrokerSubmitter
    from open_cake_ir.serialization import canonical_json_bytes
    from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
    from open_cake_ir.tasks.workloads import load_workload
    from tools.launch_task import _admit_stack,_triton_builder
    workload_path=root/'campaign/original-workload.json';workload=load_workload(workload_path)
    build_root=output/'build';build_root.mkdir()
    compiler,executor,_,compiler_ref=_admit_stack(source,build_root,workload.target,'triton')
    builder=_triton_builder(executor,workload)
    if route=='cake':
        from open_cake_ir.lab.python_reference import parse_skeleton
        text=candidate_bytes.decode('utf-8')
        payload,feedback=parse_cake_submission(text,str(candidate_path),parse_skeleton,frontend.FrontendError)
        environment=TaskOpenCakeEnvironment(compiler,builder,authority_document=authority['authoring'],
            workload=workload,case_id='primary',executor=executor)
        if feedback is not None:
            submission=CandidateSubmission.seal(environment.media_type,
                (candidate_bytes or canonical_json_bytes(text)) if candidate_path.suffix=='.json'
                else canonical_json_bytes({'python_source':text}))
            write_new(output/'build.json',{'route':route,'disposition':'rejected','feedback':feedback})
            result={'status':'rejected','route':route,'phase':phase,'feedback':feedback,
                'source_sha256':source_sha256,'candidate_sha256':submission.sha256,
                'candidate_path':candidate_relative,'completed_at_epoch':time.time()}
            write_new(output/'outcome.json',result);return result
        submission=CandidateSubmission.seal(environment.media_type,
            candidate_bytes if candidate_path.suffix=='.json' else canonical_json_bytes(payload))
    else:
        try:
            native_plan(root,candidate_path,binding)
        except (ValueError,OSError) as error:
            # The plan is the author's own input: answer it as a rejected candidate the
            # author can correct, not as a controller fault that closes the search.
            feedback={'stage':'native_plan','error':str(error),'code':'NATIVE_PLAN_INVALID',
                      'expected_fields':['schema_version','route','task','target','cake_candidate',
                                         'missing_commitment','same_contract','shares_task_deadline']}
            write_new(output/'build.json',{'route':route,'disposition':'rejected','feedback':feedback})
            result={'status':'rejected','route':route,'phase':phase,'feedback':feedback,
                'source_sha256':source_sha256,'candidate_sha256':None,
                'candidate_path':candidate_relative,'completed_at_epoch':time.time()}
            write_new(output/'outcome.json',result);return result
        starter=frontend.read_schedule(root/'campaign/candidates/starter.py').document
        lowered=compiler.lower(compiler.assess(starter))
        environment=NativeTritonEnvironment(builder,toolchain_requirements=lowered.toolchain_requirements,
            authority_document={'environment_kind':'native_triton','route_split_from':'cake'},
            workload=workload,case_id='primary')
        submission=CandidateSubmission.seal(environment.media_type,candidate_bytes)
    try:
        built=environment.build(submission)
    except Exception as error:
        result=reject_build_timeout(output,error,route=route,phase=phase,source_sha256=source_sha256,
            candidate_sha256=submission.sha256,candidate_path=candidate_relative)
        if result is None:raise
        return result
    write_new(output/'build.json',{'route':route,'disposition':built.disposition,
                                   'feedback':dict(built.feedback)})
    if built.artifact_payloads: retain(output/'build-artifacts',built.artifact_payloads)
    if built.launchable is None:
        result={'status':'rejected','route':route,'phase':phase,'feedback':dict(built.feedback),
                'source_sha256':source_sha256,
                'candidate_sha256':built.submission_sha256,
                'candidate_path':candidate_relative,'completed_at_epoch':time.time()}
        write_new(output/'outcome.json',result)
        return result
    broker=phase_broker(runtime['broker'],phase,cutoff,time.time());protocol=authority['evaluation_protocol']
    prepared=json.loads((root/'campaign/prepared-baseline.json').read_text())
    baseline=load_local_baseline(root,Path(prepared['fixed_baseline_bundle_path']))
    submitter=CommandBrokerSubmitter(command=tuple(broker['command']),workload_path=workload_path,
        workload_sha256=workload.canonical_sha256,
        protocol_sha256=sha256(canonical_json_bytes(protocol)).hexdigest(),cwd=Path(broker['cwd']),
        executor=executor,compiler_reference=compiler_ref,service_user=broker['service_user'],
        service_group=broker['service_group'],timeout_seconds=broker['timeout_seconds'],
        evaluation_protocol=protocol,baseline=baseline,workload_loader=load_workload)
    logical=BoundedBrokerEvaluator(protocol,submitter).evaluate(
        built.launchable,case_id='primary',purpose=evaluation_purpose(phase),
        admission=DeadlineAdmission(root,phase))
    attempts=[]
    for number,attempt in enumerate(logical.attempts,1):
        if attempt.artifact_payloads: retain(output/f'attempt-{number}',attempt.artifact_payloads)
        attempts.append({f.name:getattr(attempt,f.name) for f in fields(attempt)
                         if f.name not in {'artifact_payloads','receipt'}})
    write_new(output/'attempts.json',attempts)
    receipt=logical.final_receipt
    if receipt is None:
        result={'status':'no_receipt','route':route,'phase':phase,'attempts':len(attempts),
                'source_sha256':source_sha256,
                'candidate_sha256':built.submission_sha256,
                'candidate_path':candidate_relative,'completed_at_epoch':time.time()}
    else:
        retain(output/'receipt-artifacts',receipt.artifact_payloads)
        result={'status':'accepted' if receipt.correctness_passed else 'incorrect','route':route,
            'phase':phase,'correctness_passed':receipt.correctness_passed,
            'timing':None if receipt.timing is None else dict(receipt.timing),
            'source_sha256':source_sha256,
            'candidate_sha256':receipt.candidate_sha256,'candidate_path':candidate_relative,
            'completed_at_epoch':time.time()}
        write_new(output/'receipt.json',{f.name:getattr(receipt,f.name) for f in fields(receipt)
                                        if f.name!='artifact_payloads'})
    write_new(output/'outcome.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    e=sub.add_parser('evaluate');e.add_argument('--candidate',type=Path,required=True);e.add_argument('--id',required=True)
    e.add_argument('--phase',choices=('search','confirmation','attribution'),default='search')
    e.add_argument('--route',choices=('cake','native_triton'),default='cake')
    a=p.parse_args(argv);root=Path.cwd().resolve()
    if a.command=='prepare':result=prepare(root)
    else:
        from development.submission import submit
        result=submit(root,a.candidate,a.id,a.phase,a.route)
    # The author side never enters context(), so the pinned Compiler is not importable here;
    # an outcome is already plain JSON as read back from response.json.
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
