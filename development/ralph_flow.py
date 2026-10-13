"""One bounded external HMZ Ralph loop with frozen Cake tools."""
import asyncio
import json
import sys
import time
from pathlib import Path

from hmz.flows import (
    Agent,
    AgentCollection,
    EnvCollection,
    FlowContext,
    FlowParams,
    HarnessError,
    HarnessNotInstalled,
    HarnessSandboxed,
    LocalEnv,
    flow,
)

sys.path.insert(0, str(Path.cwd()))
from campaign.compiler_tools import author_context


class Agents(AgentCollection):
    agent: Agent


class Envs(EnvCollection):
    workspace: LocalEnv


@flow(agents=Agents, envs=Envs, params=FlowParams, resumable=False)
async def run(
    task: str, *, agents: Agents, envs: Envs, params: FlowParams, ctx: FlowContext
) -> None:
    """Run fresh author sessions until the owner closes search or three turns stall."""
    del params, ctx
    root = Path.cwd()
    agent = agents["agent"]
    stalled = 0
    while True:
        deadline = json.loads((root / "campaign/deadline.json").read_text())
        if (time.time() >= deadline["search_stop_at_epoch"]
                or (root / "campaign/SEARCH_CLOSE.json").exists()):
            return
        from development.statecard import render
        render(root)
        prompt = (
            task + author_context(root)
            + "\nRead campaign/management/StateCard.json and both skills named in TASK.md. "
              "Use only this task history. Token usage is accounting only. "
              "Evaluate candidates only through python3 -m development.runtime evaluate. "
              "Keep Cake and separately planned native Triton evidence distinct."
        )
        session = await agent.spawn()
        try:
            answer = await agent.run(prompt, session=session, env=envs["workspace"])
        except (HarnessNotInstalled, HarnessSandboxed):
            raise
        except HarnessError as error:
            print(f"author turn failed: {error}")
            answer = ""
        stalled = 0 if answer else stalled + 1
        if stalled >= 3:
            print("stopping: three author turns in a row returned no answer")
            return
        await asyncio.sleep(2)
