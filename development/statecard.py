"""Derived author view of this task's own retained evaluations."""
import json
from pathlib import Path

def render(root=Path.cwd()):
    rows=[]
    for path in sorted((root/'campaign/evaluations').glob('*/outcome.json')):
        try:
            row=json.loads(path.read_text());row['id']=path.parent.name;rows.append(row)
        except (OSError,ValueError): rows.append({'id':path.parent.name,'status':'unknown_unreadable'})
    value={'deadline':json.loads((root/'campaign/deadline.json').read_text()),
           'evaluations':rows,'search_close_requested':(root/'campaign/SEARCH_CLOSE.json').exists(),
           'rule':'Use only this task history. Cake and native routes remain separate.'}
    path=root/'campaign/management/StateCard.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return value
