"""One fresh HMZ author session with one inherited assignment deadline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowContext, FlowParams, LocalEnv, flow


class Agents(AgentCollection):
    agent: Agent


class Envs(EnvCollection):
    workspace: LocalEnv


class Params(FlowParams):
    pass


@flow(agents=Agents, envs=Envs, params=Params, resumable=False)
async def run(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    root = Path.cwd()
    deadline = json.loads((root / 'campaign/deadline.json').read_text())
    if time.time() >= deadline['search_stop_at_epoch']:
        raise TimeoutError('author budget is closed')
    spec = importlib.util.spec_from_file_location('selected_compiler_tools', root / 'campaign/hmz_compiler_tools.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.catalog(root)
    prompt = task + '\n' + adapter.author_context(root)
    session = await agents['agent'].spawn()
    await agents['agent'].run(prompt, session=session, env=envs['workspace'])
