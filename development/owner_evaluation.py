"""Evaluation entry admitted only as a direct child of the live task launcher."""
import argparse
import os
import sys
from pathlib import Path

from development.runtime import evaluate


def require_owner(root):
    from development.controller_entry import controller_command
    parent=Path(f'/proc/{os.getppid()}')
    command=(parent/'cmdline').read_bytes().split(b'\0')
    if command and command[-1]==b'':command.pop()
    if command!=[os.fsencode(x) for x in controller_command(root,'launch')]:
        raise PermissionError('evaluation must be a direct child of the task launcher')
    if Path(os.readlink(parent/'exe')).resolve()!=Path(sys.executable).resolve():
        raise PermissionError('task launcher executable differs')
    if Path(os.readlink(parent/'cwd')).resolve()!=root.resolve():
        raise PermissionError('task launcher workspace differs')


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--phase',choices=('search','attribution','confirmation'),required=True)
    parser.add_argument('--route',choices=('cake','native_triton'),required=True)
    args=parser.parse_args(argv);root=Path.cwd().resolve();require_owner(root)
    evaluate(root,args.candidate,args.id,args.phase,args.route)


if __name__=='__main__':main()
