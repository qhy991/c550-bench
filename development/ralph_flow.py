"""One bounded external HMZ author loop with frozen Cake tools."""
import json,sys,time
from pathlib import Path
from typing import Any
from hmz.flows import Agent,Allowance,flow
sys.path.insert(0,str(Path.cwd()))
from campaign.compiler_tools import author_context
@flow(budget=Allowance(hours=3),resumable=False)
def run(agents:tuple[Agent],task:str,state:dict[str,Any]|None=None)->None:
    (agent,)=agents;root=Path.cwd()
    while True:
        deadline=json.loads((root/'campaign/deadline.json').read_text())
        if time.time()>=deadline['search_stop_at_epoch'] or (root/'campaign/SEARCH_CLOSE.json').exists():return
        from development.statecard import render
        render(root)
        prompt=(task+author_context(root)+
          '\nRead campaign/management/StateCard.json and both skills named in TASK.md. '
          'Use only this task history. Token usage is accounting only. '
          'Evaluate candidates only through python3 -m development.runtime evaluate. '
          'Keep Cake and separately planned native Triton evidence distinct.')
        agent(prompt,suppress=True);time.sleep(2)
