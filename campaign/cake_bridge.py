"""Standalone frozen Cake Compiler adapter for xcore1000 (MetaX C500)."""
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAKE = ROOT / '.deps/cake-ir'


def _pin():
    plan = json.loads((ROOT / 'campaign/frozen-plan.json').read_text())
    binding = json.loads((ROOT / 'campaign/binding.json').read_text())
    return plan['bindings'][binding['condition']]['commit']


@functools.lru_cache(maxsize=1)
def compiler():
    sys.path.insert(0, str(CAKE / 'src'))
    from open_cake_ir.compiler import Compiler
    result = Compiler.load(CAKE, CAKE / 'compiler/revision.json')
    pin = _pin()
    if result.commit != pin:
        raise ValueError('Cake Compiler needs its clean pinned checkout at ' + pin)
    return result


def inside(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT)
    return path


def emit(schedule_path, output_dir):
    """CPU-only authoring: preserve assessment, Schedule and unedited source."""
    document = json.loads(inside(schedule_path).read_text())
    if document.get('target') != 'xcore1000' or document.get('lowering', {}).get('backend') != 'triton':
        raise ValueError('This arm is exact xcore1000 + Triton')
    engine = compiler()
    destination = inside(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'schedule.json').write_text(json.dumps(document, indent=2) + '\n')
    assessment = engine.assess(document)
    details = {
        'compiler_commit': engine.commit, 'accepted': assessment.accepted,
        'lowering_eligible': assessment.lowering_eligible,
        'findings': [f.to_dict() for f in assessment.findings],
        'guidance': [f.to_dict() for f in assessment.guidance],
        'analysis': dict(assessment.analysis),
    }
    (destination / 'assessment.json').write_text(json.dumps(details, indent=2, default=str) + '\n')
    if not assessment.lowering_eligible:
        raise ValueError('Schedule refused; see ' + str(destination / 'assessment.json'))
    lowering = engine.lower(assessment)
    (destination / 'source.py').write_text(lowering.source)
    record = {
        'compiler_commit': engine.commit, 'entry_point': lowering.route.entry_point,
        'source': str((destination / 'source.py').relative_to(ROOT)),
        'schedule': str((destination / 'schedule.json').relative_to(ROOT)),
        'assessment': str((destination / 'assessment.json').relative_to(ROOT)),
        'source_sha256': lowering.source_sha256,
    }
    (destination / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    return str((destination / 'receipt.json').relative_to(ROOT))


def verify(receipt):
    """CPU replay at artifact handoff: changing emitted source invalidates it."""
    record = json.loads(inside(receipt).read_text())
    pin = _pin()
    if record['compiler_commit'] != pin:
        raise ValueError('Compiler identity drift')
    document = json.loads(inside(record['schedule']).read_text())
    assessment = compiler().assess(document)
    lowered = compiler().lower(assessment)
    actual = inside(record['source']).read_text()
    if actual != lowered.source or record['entry_point'] != lowered.route.entry_point:
        raise ValueError('Artifact is not the unchanged Compiler emission')
    return True


@functools.lru_cache(maxsize=256)
def load(receipt):
    """Device adapter: load an already emitted artifact, no Compiler in timer."""
    record = json.loads(inside(receipt).read_text())
    pin = _pin()
    path = inside(record['source'])
    if (record['compiler_commit'] != pin
            or hashlib.sha256(path.read_bytes()).hexdigest() != record['source_sha256']):
        raise ValueError('Frozen artifact changed at device handoff')
    spec = importlib.util.spec_from_file_location('cake_' + record['source_sha256'][:16], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, record['entry_point'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['emit', 'verify', 'verify-many'])
    parser.add_argument('path', nargs='+')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if args.mode == 'emit':
        if not args.output_dir:
            parser.error('emit needs a create-only output directory')
        if len(args.path) != 1:
            parser.error('emit needs one Schedule')
        print(emit(args.path[0], args.output_dir))
    else:
        for path in args.path:
            print(path, verify(path))
