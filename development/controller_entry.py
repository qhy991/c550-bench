"""Run controller code from its pinned read-only checkout, outside author imports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]


def controller_command(root, action, *arguments):
    return [sys.executable, '-I', str(SOURCE/'development/controller_entry.py'),
            '--root', str(root), action, *arguments]


def admit(root):
    if not sys.flags.isolated:
        raise PermissionError('controller requires isolated Python imports')
    if not os.statvfs(SOURCE).f_flag & os.ST_RDONLY:
        raise PermissionError('controller source must be mounted read-only')
    binding=json.loads((root/'campaign/development-binding.json').read_text())
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if commit!=binding['adapter_commit']:
        raise ValueError('controller source differs from pinned launcher')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise ValueError('controller source is not clean')
    sys.path.insert(0,str(SOURCE))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('action',choices=('launch','prepare','evaluate'))
    args,arguments=parser.parse_known_args()
    root=args.root.resolve(strict=True);admit(root);os.chdir(root)
    if args.action=='launch':
        from development.launch import main as launch
        return launch(arguments)
    if args.action=='prepare':
        if arguments:parser.error('prepare accepts no arguments')
        from development.runtime import prepare
        prepare(root);return 0
    from development.owner_evaluation import main as evaluate
    return evaluate(arguments)


if __name__=='__main__':raise SystemExit(main())
