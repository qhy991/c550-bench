import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import time
from dataclasses import dataclass
from unittest.mock import patch

from development.binding import SKILLS, TARGET, validate
from development.prepare import prepare
from development.runtime import evaluation_purpose, load_local_baseline, native_plan, parse_cake_submission
from development.queue import DevelopmentQueue, reconcile
from development.launch import hmz_command, main as launch_main, release_verified, select_nominee
from development.submission import abandon_pending, execute, service, validate as validate_request
from development.owner_evaluation import require_owner
from development.controller_entry import SOURCE, controller_command
from development.report import combine, report as development_report

ROOT=Path(__file__).resolve().parents[1]
TASKS=json.loads((ROOT/'development/tasks.json').read_text())['tasks']


def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def repository(root,files):
    root.mkdir();subprocess.run(['git','init','-q',str(root)],check=True)
    subprocess.run(['git','-C',str(root),'config','--local','user.name','Fixture'],check=True)
    subprocess.run(['git','-C',str(root),'config','--local','user.email','fixture@example.invalid'],check=True)
    for name,text in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'commit','-qm','fixture'],check=True)
    return git(root,'rev-parse','HEAD')


class DevelopmentPreparation(unittest.TestCase):
    def inputs(self,directory):
        root=Path(directory).resolve();adapter=root/'adapter';compiler=root/'compiler'
        adapter_commit=repository(adapter,{'.gitignore':'.deps/\n'})
        tool=("import json,pathlib,sys\nroot=pathlib.Path.cwd()\n"
              "assert sys.argv[1:]==['freeze']\n"
              "b=json.loads((root/'campaign/development-binding.json').read_text())\n"
              "(root/'campaign/compiler-api.json').write_text(json.dumps({'compiler_commit':b['compiler'],'transformations':[]}))\n")
        compiler_commit=repository(compiler,{'tools/hmz_compiler_tools.py':tool,
            'skills/evolve/SKILL.md':'# evolve\n','skills/metax-c550-hardware-exploration/SKILL.md':'# metax\n'})
        source=root/'source';source.mkdir();
        for name,text in {'workload.json':'{}','starter.py':'# starter\n','run.json':'{}','baseline.json':'{}'}.items():(source/name).write_text(text)
        devices=[{'physical_device':i,'runtime_device':i,'expected_pci':f'0000:{i:02x}:00'} for i in range(8)]
        hosts={h:{'compiler_root':str(compiler),'adapter_root':str(adapter),'home':str(root/(h+'-home')),
                  'owner_root':str(root/(h+'-runs')),'devices':devices} for h in ('c550-1','c550-2')}
        assignments=[]
        for i,task in enumerate(TASKS):
            host='c550-1' if i==0 else 'c550-2'
            assignments.append({'task':task,'host':host,'root':str(Path(hosts[host]['owner_root'])/task),
                'device':devices[i%8],'source':{'workload':str(source/'workload.json'),'starter':str(source/'starter.py'),
                    'prior_run':str(source/'run.json'),'prepared_baseline':str(source/'baseline.json')}})
        plan={'schema_version':1,'kind':'c550_hmz_development_round','target':TARGET,
              'compiler_commit':compiler_commit,'adapter_commit':adapter_commit,
              'controls':{'model':'glm-5.3','reasoning_effort':'high','wall_time_seconds':10800,
                'search_seconds':9720,'confirmation_seconds':1080,'tokens':'accounting_only',
                'authors_per_device':2,'reference_access':'known_kernel_reproduction'},
              'skills':SKILLS,'hosts':hosts,'assignments':assignments,
              'unsupported':[{'task':'tinygemm','status':'unsupported_target','target':TARGET}]}
        path=root/'plan.json';path.write_text(json.dumps(plan));return root,path,plan

    def test_prepare_exposes_both_skills_hmz_catalog_and_route_split(self):
        with tempfile.TemporaryDirectory() as d:
            root,path,plan=self.inputs(d);output=root/'allocation.json'
            allocation=prepare(path,'c550-1',output,repository=ROOT)
            self.assertEqual(len(allocation['runs']),1)
            self.assertEqual(allocation['runs'][0]['home'],plan['hosts']['c550-1']['home'])
            run=Path(allocation['runs'][0]['root'])
            task=(run/'campaign/TASK.md').read_text()
            self.assertIn('$evolve',task);self.assertIn('$metax-c550-hardware-exploration',task)
            self.assertIn('separate',task.lower());self.assertIn('native Triton',task)
            self.assertTrue((run/'campaign/compiler-api.json').is_file())
            self.assertEqual(json.loads((run/'campaign/prior-run.json').read_text()),{})
            self.assertTrue((run/'.deps/cake-ir/skills/evolve/SKILL.md').is_file())
            self.assertFalse((run/'campaign/deadline.json').exists())
            self.assertFalse((run/'campaign/intake.json').exists())
            self.assertEqual(git(run,'status','--porcelain'),'')

    def test_plan_requires_all_54_and_exact_controls(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,plan=self.inputs(d)
            plan['assignments'].pop()
            with self.assertRaisesRegex(ValueError,'54-task'):
                validate(plan,ROOT)

    def test_missing_skill_stops_before_workspace_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root,path,plan=self.inputs(d)
            skill=Path(plan['hosts']['c550-1']['compiler_root'])/'skills/evolve/SKILL.md'
            skill.unlink();subprocess.run(['git','-C',str(skill.parents[2]),'add','-u'],check=True)
            subprocess.run(['git','-C',str(skill.parents[2]),'commit','-qm','remove'],check=True)
            plan['compiler_commit']=git(skill.parents[2],'rev-parse','HEAD');path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'skill'):
                prepare(path,'c550-1',root/'allocation.json',repository=ROOT)
            self.assertFalse(Path(plan['assignments'][0]['root']).exists())

    def test_native_route_requires_a_retained_own_cake_candidate_and_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();cake=root/'campaign/candidates/cake.py'
            native=root/'campaign/native-engineering/trial';native.mkdir(parents=True)
            cake.parent.mkdir(parents=True);cake.write_text('# retained Cake hypothesis\n')
            candidate=native/'candidate.json';candidate.write_text('{}')
            plan={'schema_version':1,'route':'native_triton','task':'rmsnorm','target':'xcore1002',
                  'cake_candidate':'campaign/candidates/cake.py',
                  'missing_commitment':'Cake cannot express the measured instruction choice',
                  'same_contract':True,'shares_task_deadline':True}
            (native/'plan.json').write_text(json.dumps(plan))
            self.assertEqual(native_plan(root,candidate,{'task':'rmsnorm'}),plan)
            plan['same_contract']=False;(native/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'route split'):
                native_plan(root,candidate,{'task':'rmsnorm'})

    def test_local_baseline_cannot_escape_the_task_preparation_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'campaign/baseline-preparation').mkdir(parents=True)
            foreign=root/'foreign.json';foreign.write_text('{}')
            with self.assertRaisesRegex(ValueError,'escapes'):
                load_local_baseline(root,foreign)

    def test_hmz_uses_the_flow_owned_budget_interface(self):
        command=hmz_command(Path('/opt/hmz'),'author task',9720)
        self.assertEqual(command,['/opt/hmz','exec','-f','./development/ralph_flow.py',
            '-a','agent=claude/glm-5.3:high','-p',
            'budget.duration=9720,budget.graceful=false','author task'])
        self.assertNotIn('-c',command)
        flow=(ROOT/'development/ralph_flow.py').read_text()
        self.assertIn('resumable=True',flow)
        self.assertIn('max_rounds',flow)

    def test_controller_can_pass_empty_launch_arguments(self):
        with patch('sys.argv',['controller_entry.py','--root','/task','launch']), \
             patch('development.launch.read',side_effect=FileNotFoundError('binding sentinel')):
            with self.assertRaisesRegex(FileNotFoundError,'binding sentinel'):launch_main([])

    def test_confirmation_maps_to_the_evaluators_confirmatory_purpose(self):
        self.assertEqual(evaluation_purpose('search'),'search')
        self.assertEqual(evaluation_purpose('attribution'),'attribution')
        self.assertEqual(evaluation_purpose('confirmation'),'confirmatory')

    def test_nomination_requires_one_candidate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evaluations=root/'campaign/evaluations'
            for name,phase,source in (('search','search','a'*64),('profile','attribution','b'*64)):
                path=evaluations/name;path.mkdir(parents=True)
                (path/'outcome.json').write_text(json.dumps({'phase':phase,'status':'accepted',
                    'candidate_path':'campaign/candidates/x.py','route':'cake',
                    'source_sha256':source,'candidate_sha256':'c'*64,
                    'timing':{'pooled_means_ms':{'candidate':1.0},'measurement_quality_passed':True}}))
            with patch('development.launch.ROOT',root):self.assertIsNone(select_nominee())
            row=json.loads((evaluations/'profile/outcome.json').read_text());row['source_sha256']='a'*64
            (evaluations/'profile/outcome.json').write_text(json.dumps(row))
            with patch('development.launch.ROOT',root):self.assertEqual(select_nominee()[1],'search')

    def test_release_is_scoped_to_this_task_not_the_shared_device(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();root=base/'task';other=base/'other';proc=base/'proc'
            root.mkdir();other.mkdir();(proc/'123').mkdir(parents=True)
            (proc/'123/cwd').symlink_to(other);(proc/'123/cmdline').write_bytes(b'other')
            self.assertTrue(release_verified(root,proc))
            (proc/'123/cmdline').write_bytes(str(root.parent/(root.name+'_sibling')).encode())
            self.assertTrue(release_verified(root,proc))
            (proc/'123/cwd').unlink();(proc/'123/cwd').symlink_to(root)
            self.assertFalse(release_verified(root,proc))


class _Immediate:
    def poll(self): return 0


class _QueueLauncher:
    def __init__(self): self.active=0;self.maximum=0;self.started=[]
    def start(self,run,claim,state):
        self.active+=1;self.maximum=max(self.maximum,self.active);self.started.append(run['task'])
        return _Immediate()
    def terminal(self,run,handle,code):
        self.active-=1
        return {'task':run['task'],'exit_code':code,'done':{'terminal':True},'release_verified':True}


class DevelopmentQueueTests(unittest.TestCase):
    def test_two_authors_share_one_device_without_duplicate_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=[]
            for task in ('a','b','c'):
                path=root/task;binding={'task':task,'root':str(path),'device':{'physical_device':2}}
                commit=repository(path,{'campaign/development-binding.json':json.dumps(binding)})
                runs.append({**binding,'prepared_commit':commit})
            allocation=root/'allocation.json';allocation.write_text(json.dumps({
                'kind':'c550_hmz_development_allocation','runs':runs}))
            launcher=_QueueLauncher();result=DevelopmentQueue(
                allocation,root/'state',launcher,sleeper=lambda _:None).run()
            self.assertEqual(result,{'assigned':3,'started':3,'terminal':3,'blocked_devices':[]})
            self.assertEqual(launcher.maximum,2)
            self.assertEqual(len(list((root/'state/claims').glob('*.json'))),3)
            with self.assertRaises(FileExistsError):
                DevelopmentQueue(allocation,root/'state',launcher).run()

    def test_dead_owner_reconciliation_seals_started_and_unstarted_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=[]
            for task in ('started','unstarted'):
                path=root/task;binding={'task':task,'root':str(path),'device':{'physical_device':2}}
                commit=repository(path,{'campaign/development-binding.json':json.dumps(binding)})
                runs.append({**binding,'prepared_commit':commit})
            allocation=root/'allocation.json';allocation.write_text(json.dumps({
                'kind':'c550_hmz_development_allocation','runs':runs}))
            state=root/'state'
            for name in ('claims','commands','owners','logs','terminals'):(state/name).mkdir(parents=True)
            (state/'LAUNCH.json').write_text(json.dumps({'owner_pid':99999999,'owner_start_ticks':'0'}))
            (state/'claims/started.json').write_text('{}')
            done={'terminal':True,'device_release_verified':True}
            (root/'started/campaign/DONE.json').write_text(json.dumps(done))
            result=reconcile(allocation,state)
            self.assertEqual(result['terminal'],2)
            self.assertEqual(json.loads((state/'terminals/started.json').read_text())['done'],done)
            self.assertEqual(json.loads((state/'terminals/unstarted.json').read_text())['error'],
                             'unstarted_owner_exit')


class EvaluationSubmissionTests(unittest.TestCase):
    def inputs(self,root):
        folder=root/'campaign/evaluation-requests/attempt';folder.mkdir(parents=True)
        candidate=root/'campaign/candidates/c.py';candidate.parent.mkdir();candidate.write_text('# source')
        (root/'campaign/deadline.json').write_text(json.dumps({'search_stop_at_epoch':time.time()+60}))
        request={'candidate':'campaign/candidates/c.py','id':'attempt','phase':'search','route':'cake'}
        (folder/'request.json').write_text(json.dumps(request))
        return folder,request

    def test_controller_executes_each_request_once_and_returns_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder,_=self.inputs(root);calls=[]
            def runner(root,request,env,remaining,folder):
                self.assertTrue((folder/'claimed.json').exists());calls.append(request)
                output=root/'campaign/evaluations/attempt';output.mkdir(parents=True)
                (output/'outcome.json').write_text(json.dumps({'status':'accepted'}));return 0
            service(root,{},runner);service(root,{},runner)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((folder/'response.json').read_text()),
                             {'outcome':{'status':'accepted'}})

    def test_unknown_controller_failure_closes_search_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder,_=self.inputs(root);calls=[]
            def runner(*args):calls.append(True);return 1
            service(root,{},runner);service(root,{},runner)
            self.assertEqual(len(calls),1)
            self.assertTrue((root/'campaign/SEARCH_CLOSE.json').exists())
            self.assertTrue(json.loads((folder/'response.json').read_text())['no_retry'])

    def test_author_cannot_request_confirmation_or_a_foreign_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,request=self.inputs(root)
            request['phase']='confirmation'
            with self.assertRaisesRegex(ValueError,'owns final'):validate_request(root,request)
            request['phase']='search';request['candidate']=str(Path(__file__).resolve())
            with self.assertRaisesRegex(ValueError,'outside'):validate_request(root,request)

    def test_abandoned_request_is_sealed_without_starting_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder,_=self.inputs(root)
            (folder/'request.json').unlink()
            abandon_pending(root,'owner stopped')
            self.assertEqual(json.loads((folder/'response.json').read_text()),
                             {'error':'owner stopped','no_retry':True})

    def test_owner_entry_refuses_an_ordinary_author_parent(self):
        with patch('development.owner_evaluation.Path') as path:
            path.return_value.__truediv__.return_value.read_bytes.return_value=b'python\0author.py\0'
            with self.assertRaises(PermissionError):require_owner(Path('/tmp/task'))

    def test_owner_entry_refuses_launcher_tokens_in_arbitrary_parent_arguments(self):
        with patch('development.owner_evaluation.Path') as path:
            path.return_value.__truediv__.return_value.read_bytes.return_value=(
                sys.executable.encode()+b'\0-c\0arbitrary_parent_code\0-m\0development.launch\0')
            with self.assertRaises(PermissionError):require_owner(Path('/tmp/task'))

    def test_controller_imports_ignore_task_local_evaluator_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'development').mkdir()
            (root/'development/__init__.py').write_text('')
            for module in ('runtime','owner_evaluation','submission','launch'):
                (root/f'development/{module}.py').write_text('raise RuntimeError("author replacement executed")')
            code=('import sys;sys.path.insert(0,sys.argv[1]);'
                  'from development import runtime,owner_evaluation,submission,launch;'
                  'print(runtime.__file__)')
            result=subprocess.check_output([sys.executable,'-I','-c',code,str(SOURCE)],
                                           cwd=root,text=True).strip()
            self.assertEqual(Path(result),SOURCE/'development/runtime.py')
            command=controller_command(root,'evaluate')
            self.assertEqual(command[:3],[sys.executable,'-I',str(SOURCE/'development/controller_entry.py')])

    def test_evaluation_timeout_terminates_its_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder,request=self.inputs(root)
            real_popen=subprocess.Popen
            process=[]
            def popen(*args,**kwargs):
                child=real_popen([sys.executable,'-c','import time;time.sleep(60)'],
                    stdout=kwargs['stdout'],stderr=kwargs['stderr'],start_new_session=True)
                process.append(child);return child
            with patch('development.submission.subprocess.Popen',side_effect=popen):
                with self.assertRaises(subprocess.TimeoutExpired):
                    execute(root,request,{},0.02,folder)
            self.assertIsNotNone(process[0].poll())


class DevelopmentReportTests(unittest.TestCase):
    def test_unstarted_and_failed_tasks_stay_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state';(state/'terminals').mkdir(parents=True)
            runs=[]
            for task in ('a','b'):
                workspace=root/task;workspace.mkdir();runs.append({'task':task,'root':str(workspace)})
            allocation=root/'allocation.json';allocation.write_text(json.dumps({
                'kind':'c550_hmz_development_allocation','host':'fixture',
                'compiler':'fixed-compiler','adapter_commit':'fixed-launcher','runs':runs}))
            (state/'terminals/a.json').write_text(json.dumps({'error':'unstarted_owner_exit','exit_code':None}))
            result=development_report(allocation,state)
            self.assertEqual(result['assigned'],2)
            self.assertEqual(result['execution_counts'],{'terminal':1,'unstarted':1})
            self.assertFalse(any(row['confirmation']['qualified'] for row in result['rows']))
            self.assertEqual(result['rows'][0]['terminal_error'],'unstarted_owner_exit')

    def test_partial_or_duplicate_host_views_cannot_claim_all54_coverage(self):
        host={'compiler':'fixed','launcher':'fixed','rows':[{'task':'rmsnorm'}]}
        with self.assertRaisesRegex(ValueError,'denominator'):combine([host])
        with self.assertRaisesRegex(ValueError,'duplicate'):combine([host,host])

    def test_all54_coverage_is_independent_of_success(self):
        rows=[{'task':task,'execution':'unstarted'} for task in TASKS]
        result=combine([{'compiler':'fixed','launcher':'fixed','rows':rows[:27]},
                        {'compiler':'fixed','launcher':'fixed','rows':rows[27:]}])
        self.assertTrue(result['complete_denominator'])
        self.assertFalse(result['performance_claim'])


class CandidateConstructionFeedbackTests(unittest.TestCase):
    def test_frontend_refusal_returns_localized_feedback_but_unknown_fault_raises(self):
        @dataclass
        class Location:
            filename:str='candidate.py'
            line:int=6
            column:int=40
        class FrontendRefusal(ValueError):
            code='FRONTEND_UNSUPPORTED';location=Location()
        def refused(*args,**kwargs):raise FrontendRefusal('expected a literal or a previously declared name')
        payload,feedback=parse_cake_submission('source','candidate.py',refused,FrontendRefusal)
        self.assertIsNone(payload)
        self.assertEqual(feedback['stage'],'assessment')
        self.assertEqual(feedback['source_location']['line'],6)
        def unknown(*args,**kwargs):raise RuntimeError('infrastructure failure')
        with self.assertRaisesRegex(RuntimeError,'infrastructure'):
            parse_cake_submission('source','candidate.py',unknown,FrontendRefusal)

    def test_valid_program_retains_the_program_submission_route(self):
        payload,feedback=parse_cake_submission('source','candidate.py',
            lambda *args,**kwargs:{'program_id':'own-program'},ValueError)
        self.assertEqual(payload,{'python_program_source':'source','program_id':'own-program'})
        self.assertIsNone(feedback)

if __name__=='__main__':unittest.main()
