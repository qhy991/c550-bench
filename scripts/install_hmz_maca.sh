#!/usr/bin/env bash
# Build a run-private hmz whose workspace fence admits the MetaX device node.
# hmz 0.1.0 and 0.2.1 fences grant /dev/nvidia*, /dev/dri and /dev/kfd but not /dev/mxcd,
# so a fenced author sees no MetaX device and every evaluation fails before correctness.
set -euo pipefail
source_tool=${1:?usage: install_hmz_maca.sh SOURCE_UV_TOOL_DIR [TARGET]}
target=${2:-$HOME/.cake-c500-hmz}
fence=$(cd "$source_tool" && ls lib/python3.*/site-packages/hmz/coganchor/fence/__init__.py)
version=$(sed -n 's/^Version: //p' "$source_tool"/lib/python3.*/site-packages/hmz-*.dist-info/METADATA)
case "$version" in
  0.1.0|0.2.1) ;;
  *) echo "hmz $version: the fence patch was checked against 0.1.0 and 0.2.1 only" >&2; exit 2 ;;
esac
test ! -e "$target" || { echo "$target exists; remove it to rebuild" >&2; exit 2; }
cp -a "$source_tool" "$target"
python3 - "$target/$fence" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
anchor = '        *sorted(str(one) for one in Path("/dev").glob("nvidia*")),\n'
if text.count(anchor) != 1:
    sys.exit('fence anchor not found exactly once; hmz changed')
open(path, 'w').write(text.replace(
    anchor, anchor + '        *sorted(str(one) for one in Path("/dev").glob("mxcd*")),\n'))
PY
python="$target/bin/python"
for entry in "$target/bin/hmz"; do sed -i "1s#.*#\#!$python#" "$entry"; done
"$target/bin/hmz" --version
