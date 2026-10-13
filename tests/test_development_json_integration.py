"""Exercise JSON input through the pinned Compiler without provider or device work."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from development.runtime import evaluate


@unittest.skipUnless(os.environ.get('OPEN_CAKE_COMPILER_ROOT'),
                     'set OPEN_CAKE_COMPILER_ROOT to a clean pinned Compiler checkout')
class JsonCompilerBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=Path(os.environ['OPEN_CAKE_COMPILER_ROOT']).resolve(strict=True)
        sys.path[:0]=[str(cls.source),str(cls.source/'src')]
        from open_cake_ir.compiler import Compiler,frontend
        from open_cake_ir.lab.actions import candidate_program
        from open_cake_ir.tasks.workloads import create_task
        cls.compiler=Compiler.load(cls.source,cls.source/'compiler/revision.json')
        cls.workload,starter=create_task('silu',backend='triton-metax',rows=32,columns=64)
        cls.schedule=frontend.parse(starter).document
        cls.program=candidate_program(json.dumps({'python_source':starter}).encode()).document

    def run_candidate(self,document,identity,*,compact=False,raw_text=None):
        from open_cake_ir.lab.faults import CandidateCompileRejected
        compiler=self.compiler
        class CompileProbe:
            requests=[]
            def build(self,request):
                self.requests.append(request)
                raise CandidateCompileRejected('cpu_probe_reached_toolchain',artifact_payloads={})
        probe=CompileProbe()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();campaign=root/'campaign';(campaign/'candidates').mkdir(parents=True)
            (root/'.deps').mkdir();(root/'.deps/cake-ir').symlink_to(self.source,target_is_directory=True)
            def write(name,value):(campaign/name).write_text(json.dumps(value))
            write('development-binding.json',{'compiler':compiler.commit})
            write('deadline.json',{'search_stop_at_epoch':time.time()+60})
            write('original-workload.json',self.workload)
            write('runtime.json',{})
            write('authority.json',{'evaluation_protocol':{},'authoring':{
                'input_format':'schedule_or_python_v1',
                'lowering_route':self.schedule['lowering']}})
            raw=(raw_text if raw_text is not None else json.dumps(document,
                indent=None if compact else 2,separators=(',',':') if compact else None)).encode()
            candidate=campaign/'candidates/own.json';candidate.write_bytes(raw)
            with (patch('tools.launch_task._admit_stack',return_value=(compiler,None,None,'cpu-probe')),
                  patch('tools.launch_task._triton_builder',return_value=probe)):
                result=evaluate(root,candidate,identity,'search','cake')
            retained=(campaign/'evaluations'/identity/'authored-source.json').read_bytes()
            return result,raw,retained,probe.requests

    def test_schedule_and_program_json_reach_the_existing_compiler_boundary(self):
        for document in (self.schedule,self.program):
            result,raw,retained,requests=self.run_candidate(document,'complete-json')
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['feedback']['stage'],'compile')
            self.assertEqual(result['feedback']['diagnostic'],'cpu_probe_reached_toolchain')
            self.assertEqual(len(requests),1)
            self.assertEqual(retained,raw)
            self.assertEqual(result['candidate_sha256'],result['source_sha256'])

    def test_distinct_author_bytes_keep_distinct_submission_identities(self):
        spaced,raw_spaced,_,requests=self.run_candidate(self.schedule,'spaced')
        compact,raw_compact,_,_=self.run_candidate(self.schedule,'compact',compact=True)
        self.assertEqual(json.loads(raw_spaced),json.loads(raw_compact))
        self.assertNotEqual(raw_spaced,raw_compact)
        self.assertNotEqual(spaced['candidate_sha256'],compact['candidate_sha256'])
        self.assertEqual(requests[0].candidate_sha256,spaced['source_sha256'])

    def test_original_target_lowering_abi_and_structure_gates_still_refuse(self):
        wrong_target=copy.deepcopy(self.schedule);wrong_target['target']='sm_100a'
        wrong_route=copy.deepcopy(self.schedule);wrong_route['lowering']['backend']='native_cuda'
        wrong_abi=copy.deepcopy(self.schedule)
        next(b for b in wrong_abi['buffers'] if b['mode']=='input')['shape'][0]+=1
        for document in (wrong_target,wrong_route,wrong_abi,{'reason':'applied','paths':[]}):
            result,_,_,requests=self.run_candidate(document,'invalid-json')
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['feedback']['stage'],'assessment')
            self.assertEqual(requests,[])

    def test_empty_and_malformed_sources_return_feedback_without_compilation(self):
        for text in ('','{"schedule_id":\n}'):
            result,raw,retained,requests=self.run_candidate(None,'malformed',raw_text=text)
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['feedback']['code'],'CAKE_JSON_INVALID')
            self.assertEqual(retained,raw)
            self.assertEqual(requests,[])


if __name__=='__main__':unittest.main()
