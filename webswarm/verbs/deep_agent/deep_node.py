"""Propose-verify scheduling node for the deep verb.

DeepNode manages the propose-verify loop for deeply constrained tasks. It does not search
the web directly; on each decision turn, a search child agent proposes a candidate and a
verifier child agent independently checks the claim. It calls submit_answer after the claim
survives verification or receives an acceptable weakening.
"""

import json
from copy import deepcopy
from typing import Optional

from llm_infer.llm_infer import llm_infer
from tool_env.tool_env import ToolEnv

from .prompts import (
    build_deep_system_prompt,
    is_non_existence_claim,
)
from .searcher import run_search_agent
from .verifier import run_verify_agent


# Two child-agent entry points exposed to the LLM by the deep main agent.
_call_search_agent_tool_info = {
    "type": "function",
    "function": {
        "name": "call_search_agent",
        "description": (
            "Dispatch a search sub-agent to PROPOSE a candidate answer to the given query. "
            "The sub-agent runs its own search loop and returns its best-supported answer with "
            "evidence. Use this to propose a new candidate or to gather facts you do not yet have."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Self-contained query for the search sub-agent. Include all relevant "
                        "constraints, exclusions and context. The sub-agent has no access to "
                        "your prior reasoning."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

_call_verify_agent_tool_info = {
    "type": "function",
    "function": {
        "name": "call_verify_agent",
        "description": (
            "Dispatch an INDEPENDENT verifier sub-agent to adversarially check a candidate claim. "
            "The verifier receives ONLY the claim (no proposer evidence) and runs its own "
            "investigation from scratch — this enforces source independence and prevents "
            "anchoring on the proposer's chosen sources. Returns a verdict in "
            "{refuted, weakened, survived}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": (
                        "The candidate answer / claim to verify, written as a self-contained "
                        "statement. Include all entities, time scope, and qualifiers; the "
                        "verifier has no other context."
                    ),
                },
            },
            "required": ["claim"],
        },
    },
}


class DeepNode:
    """Main-loop node for the deep verb."""

    def __init__(
        self,
        model: str,
        provider: str,
        max_steps: int,
        env_config: dict,
        root_task: Optional[str] = None,
        search_history_feedback: bool = True,
    ):
        if search_history_feedback:
            print("[DeepNode] Search history feedback is ENABLED.")

        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.env_config = env_config
        self.root_task = root_task

        # Append search/verify history to the observation so the deep agent avoids repeated paths.
        self.search_history_feedback = search_history_feedback

        # The deep main-agent prompt defines propose-verify operating rules.
        self.system_prompt = build_deep_system_prompt()

        self._build_tools()

        # reset() clears the following per-run state.
        self.messages: list[dict] = []
        self.current_observation = None
        self.current_info: dict = {}
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.trajectory: list[dict] = []
        self.all_child_results: list[dict] = []
        self._search_history: list[dict] = []  # {type: search|verify, ...}

    def _build_tools(self):
        """Build the tool list visible to the deep main agent."""
        tmp_env = ToolEnv(config=deepcopy(self.env_config))
        env_tools = {t["function"]["name"]: t for t in tmp_env.get_tools_info()}
        self.tools = [_call_search_agent_tool_info, _call_verify_agent_tool_info]
        if "submit_answer" in env_tools:
            self.tools.append(env_tools["submit_answer"])

    def _execute_call_search_agent(self, query: str) -> tuple[str, dict]:
        """Run the search child agent and compress its result into an observation for the deep main agent."""
        if not isinstance(query, str) or not query.strip():
            received = (
                f"You passed: query={query!r} (type={type(query).__name__})."
            )
            return (
                "[Error] 'query' must be a non-empty string. "
                f"{received} "
                "Re-issue call_search_agent with a self-contained query string.",
                {"error": "empty_query"},
            )

        print(f"[Deep] call_search_agent — {query[:120]}")

        try:
            raw = run_search_agent(query=query)
        except Exception as e:
            print(f"[Deep] search sub-agent error: {e}")
            return f"[Error] search sub-agent raised: {e}", {"error": str(e)}

        answer = raw.get("prediction_answer") or "(empty)"
        result = {
            "agent_kind": "search",
            "query": query,
            "answer": answer,
            "status": "completed" if raw.get("terminated") else "truncated",
            "steps": raw.get("steps", 0),
            "messages": raw.get("messages", []),
            "trajectory": raw.get("trajectory", []),
            "tool_states": raw.get("tool_states", {}),
        }

        observation = (
            f"[Search Agent Result]\n"
            f"status: {result['status']} (steps: {result['steps']})\n"
            f"query: {query}\n"
            f"---\n"
            f"{answer}"
        )

        # Add candidate search results to history for later anchor switching and verifier decisions.
        if self.search_history_feedback:
            first_line = answer.strip().split("\n")[0].strip().upper()
            is_unknown = first_line in ("ANSWER: UNKNOWN", "UNKNOWN")
            short_answer = "Unknown" if is_unknown else answer.strip().split("\n")[0].strip()[:120]
            self._search_history.append({"type": "search", "query": query, "result": short_answer})
            observation += self._render_search_history()

        print(
            f"[Deep] search done — status={result['status']} steps={result['steps']}"
        )
        return observation, result

    def _execute_call_verify_agent(self, claim: str) -> tuple[str, dict]:
        """Verify independently without proposer evidence, forcing the verifier to gather evidence from scratch."""
        if not isinstance(claim, str) or not claim.strip():
            received = (
                f"You passed: claim={claim!r} (type={type(claim).__name__})."
            )
            return (
                "[Error] 'claim' must be a non-empty, self-contained string. "
                f"{received} "
                "Re-issue call_verify_agent with a claim that includes the entity, "
                "time scope, and any qualifiers.",
                {"error": "empty_claim"},
            )

        # Nonexistence claims are unsuitable for bounded-search verification; reject them at the main-agent level.
        if is_non_existence_claim(claim):
            print(f"[Deep] verify refused (non-existence claim): {claim[:120]}")
            observation = (
                "[Verify Agent Result]\n"
                "status: refused_non_existence_claim (steps: 0)\n"
                f"claim: {claim}\n"
                "---\n"
                "VERDICT: refused\n"
                "ATTACK: Refusing to verify a non-existence assertion. Failing to find "
                "evidence of X via web search is not evidence that X does not exist; "
                "issuing SURVIVED on such a claim would only launder a search failure "
                "into a misleading verdict.\n"
                "GUIDANCE: Do not attempt to prove non-existence. Instead, switch the "
                "deep strategy: re-anchor on the rarest constraint, decompose the task "
                "into independent buckets and intersect candidate sets, or re-read the "
                "task wording for an alternative interpretation. Once you have a "
                "concrete positive candidate, send that for verification."
            )
            result = {
                "agent_kind": "verify",
                "claim": claim,
                "answer": observation,
                "status": "refused_non_existence_claim",
                "steps": 0,
                "messages": [],
                "trajectory": [],
                "tool_states": {},
            }
            # Record rejected results in history so the deep main agent does not retry similar claims.
            if self.search_history_feedback:
                short_claim = claim.strip()[:120]
                self._search_history.append({"type": "verify", "claim": short_claim, "verdict": "refused"})
                observation += self._render_search_history()
            return observation, result

        print(f"[Deep] call_verify_agent — claim={claim[:120]}")
        try:
            raw = run_verify_agent(claim=claim)
        except Exception as e:
            print(f"[Deep] verify sub-agent error: {e}")
            return f"[Error] verify sub-agent raised: {e}", {"error": str(e)}

        verdict_text = raw.get("prediction_answer") or "(empty)"
        result = {
            "agent_kind": "verify",
            "claim": claim,
            "answer": verdict_text,
            "status": "completed" if raw.get("terminated") else "truncated",
            "steps": raw.get("steps", 0),
            "messages": raw.get("messages", []),
            "trajectory": raw.get("trajectory", []),
            "tool_states": raw.get("tool_states", {}),
        }

        observation = (
            f"[Verify Agent Result]\n"
            f"status: {result['status']} (steps: {result['steps']})\n"
            f"claim: {claim}\n"
            f"---\n"
            f"{verdict_text}"
        )

        # Record the verifier verdict for later search/submit decisions.
        if self.search_history_feedback:
            verdict = self._extract_verdict(verdict_text)
            short_claim = claim.strip()[:120]
            self._search_history.append({"type": "verify", "claim": short_claim, "verdict": verdict})
            observation += self._render_search_history()

        print(f"[Deep] verify done — status={result['status']} steps={result['steps']}")
        return observation, result

    @staticmethod
    def _extract_verdict(verdict_text: str) -> str:
        """Extract a verdict keyword from verifier output."""
        import re
        m = re.search(r'VERDICT:\s*(survived|weakened|refuted|refused)', verdict_text, re.IGNORECASE)
        return m.group(1).lower() if m else "unknown"

    def _render_search_history(self) -> str:
        """Render search and verification history as text and append it to the observation."""
        if len(self._search_history) <= 1:
            return ""
        lines = ["\n\n[Investigation History]"]
        for i, entry in enumerate(self._search_history, 1):
            if entry["type"] == "search":
                lines.append(f"{i}. Search: {entry['query']} -> {entry['result']}")
            else:  # verify
                lines.append(f"{i}. Verify: {entry['claim']} -> {entry['verdict'].upper()}")
        return "\n".join(lines)

    def reset(self, task_info: dict):
        """Initialize one deep-subtask run."""
        assert "task" in task_info, "task_info missing 'task'"
        self.task_info = deepcopy(task_info)
        task_observation = task_info["task"]

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_observation},
        ]
        self.current_observation = task_observation
        self.current_info = deepcopy(task_info)
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.trajectory = [{"step": 0, "observation": task_observation}]
        self.all_child_results = []
        self._search_history: list[dict] = []

    def step(self):
        """Execute one search/verify/submit decision by the deep main agent."""
        if self.terminated or self.truncated:
            raise RuntimeError("DeepNode already finished; call reset first.")

        # 1. The deep main agent selects the next tool based on history.
        raw_response = llm_infer(
            provider=self.provider,
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )

        # 2. Deep semantics allow one action per turn; retain only the first of multiple tool calls.
        message: dict = {"role": "assistant", "content": raw_response["content"]}
        if raw_response["tool_calls"]:
            if len(raw_response["tool_calls"]) > 1:
                print("[Deep] Warning: multiple tool calls; keeping the first.")
            raw_response["tool_calls"] = [raw_response["tool_calls"][0]]
            message["tool_calls"] = [raw_response["tool_calls"][0]]
        if raw_response.get("reasoning_content"):
            message["reasoning_content"] = raw_response["reasoning_content"]
        self.messages.append(message)

        # 3. Route to the search child agent, verifier child agent, or submit_answer.
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {}

        if not raw_response["tool_calls"] and not raw_response["content"]:
            observation = "action is empty, please check the model"
            reward, terminated, truncated = 0, True, True
            info = {"action": "", "role": "user"}

        elif not raw_response["tool_calls"]:
            observation = (
                f"No tools called. You must call one of: "
                f"{[t['function']['name'] for t in self.tools]}"
            )
            info = {"action": raw_response["content"], "role": "user"}

        else:
            tool_call = raw_response["tool_calls"][0]
            tool_name = tool_call["function"]["name"]

            try:
                tool_params = tool_call["function"]["arguments"]
                if isinstance(tool_params, str):
                    tool_params = json.loads(tool_params)
            except json.JSONDecodeError as e:
                observation = f"Failed to parse tool parameters: {e}"
                info = {"action": f"{tool_name}(<parse error>)", "role": "tool"}
                self._append_tool_observation(raw_response, tool_name, observation)
                self.step_count += 1
                self.trajectory.append({
                    "step": self.step_count, "action": raw_response,
                    "observation": observation, "reward": 0,
                    "terminated": False, "truncated": False,
                })
                return observation, 0, False, False, info, info["action"]

            # A: The search child agent proposes a candidate answer.
            if tool_name == "call_search_agent":
                query = tool_params.get("query", "")
                observation, child = self._execute_call_search_agent(query)
                if "error" not in child:
                    self.all_child_results.append(child)
                info = {
                    "action": f"call_search_agent(query={json.dumps(query, ensure_ascii=False)})",
                    "role": "tool",
                }

            # B: The verifier child agent independently validates the candidate claim.
            elif tool_name == "call_verify_agent":
                claim = tool_params.get("claim", "")
                observation, child = self._execute_call_verify_agent(claim=claim)
                if "error" not in child:
                    self.all_child_results.append(child)
                info = {
                    "action": f"call_verify_agent(claim={json.dumps(claim, ensure_ascii=False)})",
                    "role": "tool",
                }

            # C: submit_answer terminates the current deep node.
            elif tool_name == "submit_answer":
                tmp_env = ToolEnv(config=deepcopy(self.env_config))
                observation, reward, terminated, truncated, info = tmp_env.step(
                    action=raw_response
                )
                info["role"] = "tool"

            else:
                observation = (
                    f"Unknown tool: {tool_name}. "
                    f"Available: {[t['function']['name'] for t in self.tools]}"
                )
                info = {"action": f"{tool_name}(?)", "role": "tool"}

        # 4. Near the step limit, remind the deep main agent to synthesize current findings.
        if not terminated and not truncated and self.step_count == self.max_steps - 2:
            observation += (
                "\nSystem warning: You have reached the maximum interaction rounds. "
                "Please synthesize current findings and call `submit_answer` now."
            )

        # 5. Update deep-node runtime state.
        action = info.get("action", "")
        self.step_count += 1
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        # 6. Add the observation as context for the next propose-verify decision.
        role = info.get("role", "user")
        if role == "tool" and raw_response.get("tool_calls"):
            self._append_tool_observation(
                raw_response,
                raw_response["tool_calls"][0]["function"]["name"],
                observation,
            )
        else:
            self.messages.append({"role": "user", "content": observation})

        # 7. Record the deep main-agent trajectory; child-agent trajectories remain in all_child_results.
        self.trajectory.append({
            "step": self.step_count,
            "action": raw_response,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        return observation, reward, terminated, truncated, info, action

    def run(self, task_info: dict) -> dict:
        """Run a complete deep subtask and return a result embeddable in the log tree."""
        self.reset(task_info=task_info)

        while not self.terminated and not self.truncated and self.step_count < self.max_steps:
            self.step()

        prediction_answer = self.current_info.get("prediction_answer", None)

        return {
            "task_info":          self.task_info,
            "prediction_answer":  prediction_answer,
            "tools":              self.tools,
            "messages":           self.messages,
            "trajectory":         self.trajectory,
            "total_reward":       0.0,
            "terminated":         self.terminated,
            "truncated":          self.truncated,
            "final_observation":  self.current_observation,
            "final_info":         self.current_info,
            "steps":              self.step_count,
            "all_child_results":  self.all_child_results,
        }

    def _append_tool_observation(self, raw_response: dict, tool_name: str, observation: str):
        tool_call_id = raw_response["tool_calls"][0]["id"]
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": observation,
        })
