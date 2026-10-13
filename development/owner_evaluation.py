"""Evaluation entry admitted only as a direct child of the live task launcher."""
import argparse
import os
from pathlib import Path

from development.runtime import evaluate


def require_owner(root):
    parent=Path(f'/proc/{os.getppid()}')
    command=(parent/'cmdline').read_bytes().split(b'\0')
    if not any(command[i:i+2]==[b'-m',b'development.launch'] for i in range(len(command)-1)):
        raise PermissionError('evaluation must be a direct child of the task launcher')
    if Path(os.readlink(parent/'cwd')).resolve()!=root.resolve():
        raise PermissionError('task launcher workspace differs')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--phase',choices=('search','attribution','confirmation'),required=True)
    parser.add_argument('--route',choices=('cake','native_triton'),required=True)
    args=parser.parse_args();root=Path.cwd().resolve();require_owner(root)
    evaluate(root,args.candidate,args.id,args.phase,args.route)


if __name__=='__main__':main()
