from pathlib import Path


def maca_protocol(base: Path, **deployment) -> dict:
    sdk = base.resolve() / 'sdk-version'
    sdk.mkdir()
    alias = base.resolve() / 'sdk'
    alias.symlink_to(sdk, target_is_directory=True)
    return {'deployment': {
        'runtime_library': str(alias / 'lib/libmcruntime.so'),
        'activity_library': str(alias / 'lib/libmcpti.so'), **deployment}}
