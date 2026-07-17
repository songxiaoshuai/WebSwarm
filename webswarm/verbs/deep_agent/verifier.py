"""Deep verifier child agent.

The verifier receives a positive candidate claim from the deep main agent, searches independently,
and returns a survived/weakened/refuted/refused verdict. It does not read evidence sources from
the search child agent, keeping the verification chain independent.
"""

import re
from datetime import datetime

from ..runtime import run_with_prompt


# System-prompt builder for the verifier child agent.

def build_verify_agent_prompt() -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"""You are a **Verifier sub-agent**. You receive a claim and must decide whether it withstands scrutiny.

Current Date: {current_date}

## What You Will and Will NOT Verify

You verify **positive, falsifiable claims**: claims that name an entity, an attribution, an event,
a value, or a relationship that can in principle be confirmed or contradicted by primary sources.

You will NOT verify **non-existence claims** of the form "no entity satisfies these constraints",
"no such X exists", "there is no record of Y", etc. Such claims are not falsifiable by web search
within a bounded budget — failing to find evidence of X is not evidence of absence — so any
verdict on them would be misleading. If the incoming claim is a non-existence claim, refuse the
task and return your refusal as the verdict (see Output Format).

## Workflow
1. **Decompose**: read the claim and list its independently checkable conditions as a numbered
   list (Condition 1, Condition 2, ...). Each condition should mention an entity, a qualifier,
   and a value/scope. This decomposition is part of the work and MUST appear in your final output.
2. **Investigate per condition**: use search/fetch_url to verify each condition independently.
   Prefer primary sources. For each condition, record the verdict (confirmed / contradicted /
   inconclusive) and the source.
3. **Aggregate**: combine the per-condition verdicts into the overall verdict (see Verdict Rules).
4. **Submit** via submit_answer in the Output Format below.

## Output Format (call submit_answer with this string)

```
VERDICT: refuted | weakened | survived | refused

CONDITIONS:
1. <condition 1 restated in your own words> — <confirmed | contradicted | inconclusive>
2. <condition 2> — ...
...

ATTACK: <one sentence — the strongest counter-argument or caveat you found; if survived, briefly
state what was strongest in favor; if refused, state why you refuse.>

EVIDENCE:
- [src: <URL>] <snippet relevant to your verdict>
- ...
```

## Verdict Rules
- **refuted**: at least one condition is contradicted by a primary source, OR a stronger
  alternative answer fits the constraints better.
- **weakened**: the claim is partially correct but must be narrowed (wrong year, wrong scope,
  over-generalization, etc.). Use this when most conditions confirm but at least one fails.
- **survived**: every condition is independently confirmed, or none could be contradicted after
  honest checking with primary sources.
- **refused**: the claim is a non-existence assertion or otherwise not falsifiable; you decline
  to issue a confidence-ranked verdict on it.

## Rules
- Mandatory tool use: call a tool every turn.
- Do NOT rely on internal knowledge. Every judgement must be backed by retrieved sources.
- Do NOT rubber-stamp. A SURVIVED that came from one cursory query is a failure.
- The CONDITIONS section MUST appear in your final answer — it is how the orchestrator audits
  what you actually checked.
"""


def run_verify_agent(
    claim: str,
) -> dict:
    """Assemble the verifier task and run the verifier child agent through run_with_prompt.

    Do not precheck is_non_existence_claim; that is the responsibility of the DeepNode
    orchestration layer. Return the raw run_with_prompt result dict.
    """
    verifier_task = (
        f"CLAIM TO VERIFY:\n{claim.strip()}\n\n"
        f"Investigate independently and return a verdict per your instructions. "
        f"You have NOT been given the proposer's sources; find your own."
    )
    return run_with_prompt(
        system_prompt=build_verify_agent_prompt(),
        task=verifier_task,
    )


# Nonexistence claims cannot be established well with bounded web search, so filter them in DeepNode first.

_NEGATIVE_CLAIM_RX = re.compile(
    r"\b(?:there\s+is|there\s+exists|there\s+are)\s+no\b"
    r"|\bno\s+(?:such|professional|known|documented|verified|recorded|public)\b"
    r"|\bdoes\s+not\s+exist\b|\bdo\s+not\s+exist\b"
    r"|\bnever\s+(?:existed|occurred|happened)\b"
    r"|\bno\s+\w+(?:\s+\w+){0,8}?\s+(?:exists|existed|matches|meets|satisfies|has|have|had)\b"
    r"|\b(?:cannot|could\s+not|can\s+not)\s+(?:be\s+)?(?:identify|identified|find|found|locate|located)(?:\s+any)?\b"
    r"|\bnone\s+(?:of\s+\w+\s+)?(?:exists|exist|match|matches|meet|meets)\b",
    re.I,
)


def is_non_existence_claim(claim: str) -> bool:
    if not isinstance(claim, str) or not claim.strip():
        return False
    return bool(_NEGATIVE_CLAIM_RX.search(claim))
