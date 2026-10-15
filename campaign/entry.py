"""Isolated entry point for trusted assignment supervisors and device assays."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('supervise', 'assay', 'reference', 'release'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if not sys.flags.isolated:
        raise PermissionError('trusted entry requires Python isolated imports')
    if not os.statvfs(SOURCE).f_flag & os.ST_RDONLY:
        raise PermissionError('trusted Bench code must be mounted read-only')
    sys.path.insert(0, str(SOURCE))
    from campaign.binding import clean_checkout, read_object
    binding = read_object(args.root / 'campaign/binding.json')
    clean_checkout(SOURCE, binding['bench_commit'])
    if args.action in ('reference', 'assay'):
        if args.request is None:
            parser.error('trusted assay requires a request')
        from campaign.processes import trusted_attempt_environment
        os.environ.update(trusted_attempt_environment(dict(os.environ), args.request.parent))
    if args.action == 'release':
        if args.request is None:
            parser.error('release requires a create-only observation path')
        from campaign.live import observe_release
        observe_release(args.root, args.request)
    elif args.action == 'reference':
        if args.request is None:
            parser.error('reference requires a trusted bounded request')
        from campaign.reference import qualify_reference
        qualify_reference(args.root, read_object(args.request), args.request.parent)
    elif args.action == 'assay':
        if args.request is None:
            parser.error('assay requires a trusted request')
        from campaign.assay import evaluate
        request = read_object(args.request)
        evaluate(args.root, request, args.request.parent)
    else:
        if args.request is not None:
            parser.error('supervisor does not accept a request override')
        from campaign.supervisor import supervise
        supervise(args.root, args.state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
