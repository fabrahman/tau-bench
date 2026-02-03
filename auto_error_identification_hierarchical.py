"""
Hierarchical Error Analysis for Automated Error Identification
Example command: python auto_error_identification.py   --model gpt-4o-mini   --platform openai   --results-path ../general-tool-use/error_identification/model_outputs/qwen3_8b_airline.json   --output-path test_hierarchical_1example.json   --hierarchical   --max-num-failed-results 1   --max-concurrency 1 --env airline
"""


import json
import argparse
from enum import Enum
from pydantic import BaseModel
from tau_bench.model_utils import default_api_from_args, API
from tau_bench.envs.airline.tasks_test import TASKS as AIRLINE_TASKS
from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS
from tau_bench.model_utils.args import api_parser
from tau_bench.types import Task, Action
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

def get_args() -> argparse.Namespace:
    parser = api_parser()
    parser.add_argument("--env", type=str, required=False, choices=["airline", "retail"], help="The environment that the original trajectories are from (optional if task info is embedded in results)")
    parser.add_argument("--results-path", type=str, required=True, help="Path to the results file")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent API calls")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--max-num-failed-results", "-n", type=int, help="Maximum number of failed results to analyze")
    parser.add_argument("--hierarchical", action="store_true", help="Enable hierarchical error analysis (more detailed but slower)")
    return parser.parse_args()

class OriginalResult(BaseModel):
    task_id: int
    user_instruction: str
    traj: List[Dict[str, Any]]
    ground_truth_actions: List[Action]
    ground_truth_outputs: List[str]

class FaultAuthor(Enum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"

class FaultAssignmentResult(BaseModel):
    task_id: int
    author: FaultAuthor
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "author": self.author.value,
            "description": self.description,
        }

class FaultType(Enum):
    CALLED_WRONG_TOOL = "called_wrong_tool"
    USED_WRONG_TOOL_ARGUMENT = "used_wrong_tool_argument"
    GOAL_PARTIALLY_COMPLETED = "goal_partially_completed"
    OTHER = "other"

class FaultTypeResult(BaseModel):
    task_id: int
    fault_type: FaultType
    description: str

    def model_dump(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fault_type": self.fault_type.value,
            "description": self.description,
        }

class GradingStrategy(Enum):
    ACTIONS = "actions"
    OUTPUTS = "outputs"

# ============================================================================
# Hierarchical Error Analysis Classes
# ============================================================================

class ErrorNode(BaseModel):
    """A node in the hierarchical error tree"""
    level: str  # "root_cause", "high_level", "mid_level", "low_level"
    category: str
    description: str
    children: List['ErrorNode'] = []
    evidence: List[str] = []
    turn_indices: List[int] = []

    def model_dump(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "category": self.category,
            "description": self.description,
            "children": [child.model_dump() for child in self.children],
            "evidence": self.evidence,
            "turn_indices": self.turn_indices,
        }


class ErrorChain(BaseModel):
    """A complete error chain from root cause to low-level details"""
    root_cause: str
    priority: str  # "critical", "major", "minor"
    chain: ErrorNode
    summary: str  # One-sentence summary of this error

    def model_dump(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "priority": self.priority,
            "summary": self.summary,
            "chain": self.chain.model_dump(),
        }

class HierarchicalErrorTaxonomy:
    """
    Multi-level error taxonomy

    Level 1 (Root Cause): WHY did the failure happen fundamentally?
    Level 2 (High-Level): WHAT type of failure occurred?
    Level 3 (Mid-Level): HOW did it manifest?
    Level 4 (Low-Level): WHAT specifically went wrong?
    """

    TAXONOMY = {
        "root_cause": {
            "policy_violation": {
                "description": "Agent violated explicit system policy or rules",
                "children": ["mandatory_step_skipped", "forbidden_action_taken", "constraint_broken"]
            },
            "capability_limitation": {
                "description": "Agent fundamentally cannot perform required operation",
                "children": ["missing_tool", "tool_insufficient", "constraint_violation"]
            },
            "knowledge_gap": {
                "description": "Agent lacks domain knowledge or understanding",
                "children": ["domain_knowledge", "procedural_knowledge", "tool_usage_knowledge"]
            },
            "reasoning_failure": {
                "description": "Agent fails to reason correctly despite having information",
                "children": ["logical_error", "planning_error", "inference_error", "priority_error"]
            },
            "context_management": {
                "description": "Agent loses or mismanages conversation/state context",
                "children": ["context_loss", "context_confusion", "state_tracking_error"]
            },
            "specification_issue": {
                "description": "Problem with task specification or tool definitions",
                "children": ["ambiguous_instruction", "conflicting_requirements", "unclear_tool_schema"]
            }
        },
        "high_level": {
            "policy_noncompliance": {
                "description": "Agent violated mandatory procedures or rules",
                "children": ["skipped_confirmation", "unauthorized_action", "security_violation"]
            },
            "goal_misunderstanding": {
                "description": "Agent misunderstood what needs to be accomplished",
                "children": ["wrong_goal", "incomplete_goal", "conflated_goals"]
            },
            "execution_error": {
                "description": "Agent understood goal but executed incorrectly",
                "children": ["wrong_action", "wrong_sequence", "incomplete_execution"]
            },
            "communication_failure": {
                "description": "Agent provided incorrect/misleading information to user",
                "children": ["hallucinated_info", "missing_info", "contradictory_info"]
            },
            "state_management_error": {
                "description": "Agent failed to track or update state correctly",
                "children": ["lost_information", "incorrect_state", "state_desync"]
            },
            "recovery_failure": {
                "description": "Agent encountered error but failed to recover",
                "children": ["no_recovery_attempt", "incorrect_recovery", "repeated_failure"]
            }
        },
        "mid_level": {
            "tool_selection_error": {
                "description": "Wrong tool chosen for the task",
                "children": ["completely_wrong_tool", "suboptimal_tool", "hallucinated_tool"]
            },
            "parameter_error": {
                "description": "Incorrect tool parameters used",
                "children": ["wrong_value", "missing_parameter", "extra_parameter", "wrong_type"]
            },
            "sequencing_error": {
                "description": "Correct actions in wrong order",
                "children": ["prerequisite_skipped", "reversed_order", "premature_action"]
            },
            "hallucination": {
                "description": "Agent made up information not in context",
                "children": ["hallucinated_data", "hallucinated_result", "hallucinated_capability"]
            },
            "premature_termination": {
                "description": "Agent stopped before completing task",
                "children": ["false_completion", "gave_up", "timeout"]
            },
            "circular_behavior": {
                "description": "Agent repeats same action without progress",
                "children": ["exact_repetition", "near_repetition", "stuck_in_loop"]
            }
        },
        "low_level": {
            "wrong_tool_called": {
                "description": "Specific tool call was incorrect",
                "examples": ["Called 'delete_booking' instead of 'modify_booking'"]
            },
            "missing_required_parameter": {
                "description": "Required parameter omitted from tool call",
                "examples": ["Called search_flights without 'destination'"]
            },
            "invalid_parameter_value": {
                "description": "Parameter value doesn't meet constraints",
                "examples": ["Used date '2023-02-30' which doesn't exist"]
            },
            "incorrect_parameter_type": {
                "description": "Parameter has wrong type",
                "examples": ["Passed string '5' instead of integer 5"]
            },
            "wrong_timing": {
                "description": "Correct tool called at wrong time",
                "examples": ["Called confirm_booking before selecting flight"]
            }
        }
    }

class HierarchicalErrorAnalyzer:
    """Performs hierarchical error analysis on failed trajectories"""

    def __init__(self, api: API):
        self.api = api
        self.taxonomy = HierarchicalErrorTaxonomy.TAXONOMY

    def analyze_hierarchical(
        self,
        result: OriginalResult,
    ) -> List[ErrorChain]:
        """
        Perform top-down hierarchical error analysis
        Returns a list of error chains (one per distinct root cause)
        """
        grading_strategy = GradingStrategy.OUTPUTS if len(result.ground_truth_outputs) > 0 else GradingStrategy.ACTIONS

        context = display_context(
            result.user_instruction,
            result.ground_truth_actions,
            result.ground_truth_outputs,
            result.traj
        )

        # Step 1: Identify ALL distinct root causes
        root_causes = self._identify_all_root_causes(context, result.traj)

        # Step 2: For each root cause, build complete error chain
        error_chains = []
        for root_cause_category in root_causes:
            chain = self._build_error_chain(context, result.traj, root_cause_category)
            error_chains.append(chain)

        # Step 3: Sort by priority (critical > major > minor)
        priority_order = {"critical": 0, "major": 1, "minor": 2}
        error_chains.sort(key=lambda x: priority_order[x.priority])

        return error_chains

    def _build_error_chain(
        self,
        context: str,
        traj: List[Dict],
        root_cause_category: str
    ) -> ErrorChain:
        """Build complete error chain for a specific root cause"""

        # Level 1: Root Cause
        root_cause = self._analyze_specific_root_cause(context, traj, root_cause_category)

        # Level 2: High-Level Error Type
        high_level = self._analyze_high_level(context, traj, root_cause)

        # Level 3: Mid-Level Manifestation
        mid_level = self._analyze_mid_level(context, traj, high_level)

        # Level 4: Low-Level Specifics
        low_level = self._analyze_low_level(context, traj, mid_level)

        # Build full tree structure
        low_level.children = []
        mid_level.children = [low_level]
        high_level.children = [mid_level]
        root_cause.children = [high_level]

        # Determine priority
        priority = self._determine_priority(root_cause_category)

        # Generate summary
        summary = self._generate_chain_summary(root_cause, low_level)

        return ErrorChain(
            root_cause=root_cause_category,
            priority=priority,
            chain=root_cause,
            summary=summary
        )

    def _identify_all_root_causes(self, context: str, traj: List[Dict]) -> List[str]:
        """Identify ALL distinct root causes in the trajectory"""

        prompt = """Identify ALL DISTINCT root causes in this failed trajectory.

A trajectory may have MULTIPLE INDEPENDENT errors. List ALL that apply:

1. policy_violation: Violated explicit system policy rules
2. capability_limitation: Lacks tools/features to succeed
3. knowledge_gap: Lacks domain knowledge
4. reasoning_failure: Has info but reasons incorrectly
5. context_management: Lost track of context/state
6. specification_issue: Task spec itself problematic

INSTRUCTIONS:
- List ALL distinct root causes you find (comma-separated)
- If trajectory has: policy violation + missing parameter + wrong value, list all 3
- Order by severity (most critical first)

FORMAT: Return comma-separated list, e.g., "policy_violation, reasoning_failure" or just "reasoning_failure"

YOUR ANSWER (comma-separated root causes):"""

        # Use generate instead of classify for multi-select
        response = self.api.generate(
            instruction=prompt,
            text=context
        )

        # Parse comma-separated response
        root_causes_str = response.strip().strip('"').strip("'")
        root_cause_list = [rc.strip() for rc in root_causes_str.split(',')]

        # Filter valid root causes
        valid_root_causes = list(self.taxonomy["root_cause"].keys())
        filtered = [rc for rc in root_cause_list if rc in valid_root_causes]

        # If none found or empty, fall back to single analysis
        if not filtered:
            return [self._analyze_root_cause_fallback(context, traj)]

        # Limit to top 3 most critical
        return filtered[:3]

    def _analyze_root_cause_fallback(self, context: str, traj: List[Dict]) -> str:
        """Fallback: identify single most critical root cause"""
        options = list(self.taxonomy["root_cause"].keys())
        option_descriptions = [
            self.taxonomy["root_cause"][opt]["description"]
            for opt in options
        ]

        prompt = """Identify the MOST CRITICAL root cause."""

        classification = self.api.classify(
            instruction=prompt,
            text=context,
            options=option_descriptions
        )

        return options[classification]

    def _analyze_specific_root_cause(
        self,
        context: str,
        traj: List[Dict],
        root_cause_category: str
    ) -> ErrorNode:
        """Analyze a specific root cause category"""

        # Generate explanation for this specific root cause
        explanation = self.api.generate(
            instruction=f"""Explain how "{root_cause_category}" occurred in this trajectory (max 2-3 sentences).

FOCUS ONLY ON: {root_cause_category}
Ignore other errors - focus on this specific root cause.

FORMAT:
- What specific policy/rule/constraint was violated OR what capability was missing
- Quote specific evidence from system policy or user instruction
- State the consequence

EXAMPLE for policy_violation: "Agent violated system policy requiring explicit user confirmation before database updates. Policy states: 'you must list action details and obtain explicit user confirmation (yes) to proceed.' Agent called book_reservation without obtaining 'yes' confirmation."

YOUR EXPLANATION for {root_cause_category}:""",
            text=context
        )

        evidence, turn_indices = self._extract_evidence_simple(traj)

        return ErrorNode(
            level="root_cause",
            category=root_cause_category,
            description=explanation.strip(),
            evidence=evidence,
            turn_indices=turn_indices
        )

    def _determine_priority(self, root_cause: str) -> str:
        """Determine priority level based on root cause"""
        # Critical: Policy violations, capability limitations
        if root_cause in ["policy_violation", "capability_limitation"]:
            return "critical"
        # Major: Knowledge gaps, reasoning failures
        elif root_cause in ["knowledge_gap", "reasoning_failure"]:
            return "major"
        # Minor: Context management, specification issues
        else:
            return "minor"

    def _generate_chain_summary(self, root: ErrorNode, leaf: ErrorNode) -> str:
        """Generate one-sentence summary of error chain"""
        return f"{root.category} → {leaf.category}: {leaf.description[:80]}..."

    def _analyze_root_cause(self, context: str, traj: List[Dict]) -> ErrorNode:
        """Identify fundamental reason for failure"""

        options = list(self.taxonomy["root_cause"].keys())
        option_descriptions = [
            self.taxonomy["root_cause"][opt]["description"]
            for opt in options
        ]

        prompt = """Identify the PRIMARY ROOT CAUSE - the MOST CRITICAL fundamental reason WHY this failure occurred.

IMPORTANT: If multiple errors exist, select THE MOST CRITICAL ONE. The subsequent analysis will trace this single error path.

PRIORITY CHECK (check these first):
1. POLICY VIOLATION: Did agent violate explicit system policy rules? (e.g., skipped mandatory confirmation, took forbidden action) - HIGHEST PRIORITY
2. CAPABILITY LIMITATION: Does agent lack tools/features to succeed?
3. KNOWLEDGE GAP: Does agent lack domain knowledge?
4. REASONING FAILURE: Did agent have info but reason incorrectly?
5. CONTEXT MANAGEMENT: Did agent lose track of context/state?
6. SPECIFICATION ISSUE: Was task spec itself problematic?

Select the SINGLE MOST CRITICAL root cause."""

        classification = self.api.classify(
            instruction=prompt,
            text=context,
            options=option_descriptions
        )

        selected_category = options[classification]

        # Generate concise explanation
        explanation = self.api.generate(
            instruction=f"""Explain why root cause is "{selected_category}".

FORMAT (max 2-3 sentences):
- What policy/rule/constraint was violated OR what capability was missing
- Quote specific evidence from system policy or user instruction
- State the consequence

EXAMPLE: "Agent violated system policy requiring explicit user confirmation before database updates. Policy states: 'you must list action details and obtain explicit user confirmation (yes) to proceed.' Agent called book_reservation without obtaining 'yes' confirmation."

YOUR EXPLANATION:""",
            text=context
        )

        # Extract evidence
        evidence, turn_indices = self._extract_evidence_simple(traj)

        return ErrorNode(
            level="root_cause",
            category=selected_category,
            description=explanation.strip(),
            evidence=evidence,
            turn_indices=turn_indices
        )

    def _analyze_high_level(
        self,
        context: str,
        traj: List[Dict],
        parent: ErrorNode
    ) -> ErrorNode:
        """Identify high-level error type, conditioned on root cause"""

        options = list(self.taxonomy["high_level"].keys())
        option_descriptions = [
            self.taxonomy["high_level"][opt]["description"]
            for opt in options
        ]

        prompt = f"""ROOT CAUSE: "{parent.category}" - {parent.description[:150]}

CRITICAL: Identify the high-level failure that was DIRECTLY CAUSED BY this root cause.
Do NOT identify a different unrelated error. The high-level error must be a CONSEQUENCE of the root cause.

EXAMPLES OF CAUSAL CONNECTIONS:
- policy_violation (root) → policy_noncompliance (high-level)
- reasoning_failure (root) → goal_misunderstanding (high-level)
- reasoning_failure (root) → execution_error (high-level)
- knowledge_gap (root) → execution_error (high-level)

Given the root cause "{parent.category}", what high-level failure did it CAUSE?

OPTIONS:
- policy_noncompliance: Violated mandatory procedures/rules
- goal_misunderstanding: Misunderstood what to accomplish
- execution_error: Understood goal but executed wrong
- communication_failure: Gave user incorrect/misleading info
- state_management_error: Failed to track state correctly
- recovery_failure: Encountered error but failed to recover

Select the high-level failure that was CAUSED BY the root cause."""

        classification = self.api.classify(
            instruction=prompt,
            text=context,
            options=option_descriptions
        )

        selected_category = options[classification]

        explanation = self.api.generate(
            instruction=f"""Describe how "{selected_category}" resulted FROM the root cause "{parent.category}" (max 2 sentences).

CRITICAL: Your description must show CAUSAL CONNECTION between root cause and high-level failure.

FORMAT: [How root cause led to this failure] + [Specific example]

EXAMPLES:
- "Agent's policy violation (no confirmation) led to noncompliant workflow. Proceeded directly to book_reservation without asking user to confirm details."
- "Agent's reasoning failure about time constraints led to wrong tool selection. Called search_direct_flight when one-stop flight was needed for 11am constraint."

ROOT CAUSE: "{parent.category}" - {parent.description[:100]}

YOUR DESCRIPTION (show causal connection):""",
            text=context
        )

        evidence, turn_indices = self._extract_evidence_simple(traj)

        return ErrorNode(
            level="high_level",
            category=selected_category,
            description=explanation.strip(),
            evidence=evidence,
            turn_indices=turn_indices
        )

    def _analyze_mid_level(
        self,
        context: str,
        traj: List[Dict],
        parent: ErrorNode
    ) -> ErrorNode:
        """Identify specific error manifestation"""

        options = list(self.taxonomy["mid_level"].keys())
        option_descriptions = [
            self.taxonomy["mid_level"][opt]["description"]
            for opt in options
        ]

        prompt = f"""High-level failure: "{parent.category}" - {parent.description[:150]}

CRITICAL: Identify HOW this specific high-level failure MANIFESTED.
Do NOT identify a different unrelated error. The mid-level error must be the SPECIFIC WAY the high-level error appeared.

EXAMPLES OF CAUSAL MANIFESTATION:
- policy_noncompliance (high) → sequencing_error (mid): skipped confirmation step
- goal_misunderstanding (high) → tool_selection_error (mid): picked wrong tool for goal
- execution_error (high) → parameter_error (mid): wrong parameters in tool call
- communication_failure (high) → hallucination (mid): claimed X when actually Y

Given "{parent.category}", HOW did this specifically manifest?

OPTIONS:
- tool_selection_error: Wrong tool chosen
- parameter_error: Wrong tool parameters (missing, incorrect value, wrong type)
- sequencing_error: Right actions, wrong order (skipped step)
- hallucination: Made up information (claimed X happened when it didn't)
- premature_termination: Stopped before completion
- circular_behavior: Repeated same action

Select the manifestation of "{parent.category}"."""

        classification = self.api.classify(
            instruction=prompt,
            text=context,
            options=option_descriptions
        )

        selected_category = options[classification]

        explanation = self.api.generate(
            instruction=f"""Describe how "{selected_category}" was the manifestation of "{parent.category}" (max 2 sentences).

CRITICAL INSTRUCTIONS:
1. For parameter_error: You MUST do detailed comparison:
   - Look at GROUND TRUTH ACTIONS section
   - Look at ACTUAL TRAJECTORY tool calls
   - Compare EVERY parameter (arrays, nested objects, values)
   - Report which parameter(s) are wrong/missing/extra

2. For hallucination: Compare agent's final message vs actual tool call results

3. For sequencing_error: Identify which specific step was skipped

FORMAT: [Specific technical detail] + [What it should be]

EXAMPLES:
- "Missing second flight segment HAT039 in flights array. Ground truth has [{{'flight_number': 'HAT136'}}, {{'flight_number': 'HAT039'}}] but actual only has [{{' flight_number': 'HAT136'}}]."
- "Policy noncompliance manifested as skipping confirmation step. Proceeded to book_reservation without listing details for user approval."

HIGH-LEVEL ERROR: "{parent.category}" - {parent.description[:100]}

YOUR DESCRIPTION (be specific with parameter names and values):""",
            text=context
        )

        evidence, turn_indices = self._extract_evidence_simple(traj)

        return ErrorNode(
            level="mid_level",
            category=selected_category,
            description=explanation.strip(),
            evidence=evidence,
            turn_indices=turn_indices
        )

    def _analyze_low_level(
        self,
        context: str,
        traj: List[Dict],
        parent: ErrorNode
    ) -> ErrorNode:
        """Identify specific technical error"""

        options = list(self.taxonomy["low_level"].keys())
        option_descriptions = [
            self.taxonomy["low_level"][opt]["description"]
            for opt in options
        ]

        prompt = f"""Mid-level error: "{parent.category}" - {parent.description[:150]}

CRITICAL: Identify the PRECISE TECHNICAL DETAILS of this mid-level error.
Do NOT identify a different error. The low-level error must be the SPECIFIC IMPLEMENTATION of the mid-level error.

EXAMPLES OF SPECIFICITY:
- parameter_error (mid) → missing_required_parameter (low): specific parameter name that's missing
- parameter_error (mid) → invalid_parameter_value (low): specific parameter with wrong value
- tool_selection_error (mid) → wrong_tool_called (low): specific tool names (wrong vs correct)
- sequencing_error (mid) → wrong_timing (low): specific step that was done at wrong time

Given "{parent.category}", what are the PRECISE TECHNICAL DETAILS?"""

        classification = self.api.classify(
            instruction=prompt,
            text=context,
            options=option_descriptions
        )

        selected_category = options[classification]

        explanation = self.api.generate(
            instruction=f"""Provide PRECISE TECHNICAL DETAILS (1-2 sentences max).

MANDATORY STEPS:
1. Look at GROUND TRUTH ACTION SEQUENCE - copy exact parameter names and values
2. Look at ACTUAL TRAJECTORY tool calls - find the corresponding tool call
3. Compare parameter-by-parameter - identify EXACT difference
4. Report with actual JSON/dict syntax showing before/after

FORMAT: [Tool.parameter with wrong value] + [Exact JSON showing correct value from ground truth]

EXAMPLES:
- "book_reservation called with flights=[{{'flight_number': 'HAT136', 'date': '2024-05-20'}}]. Ground truth requires flights=[{{'flight_number': 'HAT136', 'date': '2024-05-20'}}, {{'flight_number': 'HAT039', 'date': '2024-05-20'}}]"
- "search_direct_flight(origin='JFK', destination='SEA') called. Should be search_one_stop_flight with same parameters per ground truth."
- "payment_methods=[{{'payment_id': 'certificate_7504069', 'amount': 250}}]. Ground truth requires payment_methods=[{{'payment_id': 'certificate_4856383', 'amount': 100}}, {{'payment_id': 'credit_card_4421486', 'amount': 52}}]"

MID-LEVEL ERROR: "{parent.category}" - {parent.description[:100]}

YOUR DESCRIPTION (use exact JSON notation from ground truth):""",
            text=context
        )

        evidence, turn_indices = self._extract_evidence_simple(traj)

        return ErrorNode(
            level="low_level",
            category=selected_category,
            description=explanation.strip(),
            evidence=evidence,
            turn_indices=turn_indices
        )

    def _extract_evidence_simple(self, traj: List[Dict]) -> tuple[List[str], List[int]]:
        """Simple evidence extraction - returns first few assistant messages and their indices"""
        evidence = []
        turn_indices = []

        for i, turn in enumerate(traj):
            if turn["role"] == "assistant" and len(evidence) < 2:
                content = turn.get("content", "")
                if content:
                    evidence.append(content[:200] + "..." if len(content) > 200 else content)
                    turn_indices.append(i)

        return evidence, turn_indices

    def visualize_tree(self, root: ErrorNode, indent: int = 0) -> str:
        """Create text visualization of error tree"""

        prefix = "  " * indent
        icons = ["🔴", "├─", "├─", "└─"]
        icon = icons[min(indent, len(icons) - 1)]

        output = f"{prefix}{icon} [{root.level}] {root.category}\n"
        output += f"{prefix}   Description: {root.description[:150]}...\n"

        if root.turn_indices:
            output += f"{prefix}   Turns: {root.turn_indices}\n"

        if indent < 3:  # Add separator between levels
            output += f"{prefix}\n"

        for child in root.children:
            output += self.visualize_tree(child, indent + 1)

        return output

    def visualize_chains(self, chains: List[ErrorChain]) -> str:
        """Create text visualization of multiple error chains"""

        output = f"\n{'='*80}\n"
        output += f"FOUND {len(chains)} DISTINCT ERROR CHAIN(S)\n"
        output += f"{'='*80}\n\n"

        for i, chain in enumerate(chains, 1):
            priority_icon = {
                "critical": "🔥",
                "major": "⚠️",
                "minor": "ℹ️"
            }
            icon = priority_icon.get(chain.priority, "")

            output += f"\n{icon} ERROR CHAIN #{i} [{chain.priority.upper()}]\n"
            output += f"{'─'*80}\n"
            output += f"Summary: {chain.summary}\n"
            output += f"{'─'*80}\n"
            output += self.visualize_tree(chain.chain)
            output += "\n"

        return output

def context_description(grading_strategy: GradingStrategy) -> str:
    if grading_strategy == GradingStrategy.ACTIONS:
        return """CONTEXT PROVIDED:
1. SYSTEM POLICY (in trajectory's system message): Mandatory rules agent must follow
2. USER INSTRUCTION: Task requirements, preferences, constraints
3. GROUND TRUTH ACTIONS: Expected correct tool calls with parameters
4. ACTUAL TRAJECTORY: What agent actually did

KEY CHECKS:
- Compare GROUND TRUTH vs ACTUAL tool calls (which parameters differ/missing?)
- Check if agent violated SYSTEM POLICY rules
- Check if agent respected USER INSTRUCTION constraints/preferences
- Check if agent's final message matches what was actually done (hallucinations?)"""
    return """CONTEXT PROVIDED:
1. SYSTEM POLICY (in trajectory's system message): Mandatory rules agent must follow
2. USER INSTRUCTION: Task requirements, preferences, constraints
3. REQUIRED OUTPUTS: What agent should communicate to user
4. ACTUAL TRAJECTORY: What agent actually did/said

KEY CHECKS:
- Did agent violate SYSTEM POLICY?
- Did agent provide all REQUIRED OUTPUTS?
- Did agent respect USER INSTRUCTION constraints?"""

def display_traj(traj: List[Dict[str, Any]]) -> str:
    if len(traj) == 0:
        raise ValueError("Trajectory is empty")
    stripped_traj = [item for item in traj if item["role"] != "system"]
    return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in stripped_traj])

def display_actions(actions: List[Action]) -> str:
    return json.dumps([action.model_dump() for action in actions], indent=4)

def display_context(user_instruction: str, ground_truth_actions: List[Action], ground_truth_outputs: List[str], trajectory: List[Dict[str, Any]]) -> str:
    traj_display = display_traj(trajectory)
    context = f"""----- start user instruction -----
{user_instruction}
----- end user instruction -----"""
    if len(ground_truth_outputs) > 0:
        context += f"""

----- start required outputs -----
{ground_truth_outputs}
----- end required outputs -----"""
    else:
        context += f"""

----- start ground truth action sequence -----
{display_actions(ground_truth_actions)}
----- end ground truth action sequence -----

----- start trajectory -----
{traj_display}
----- end trajectory -----\n"""
    return context

def fault_assignment_analysis(api: API, results: List[OriginalResult], max_concurrency: int) -> List[FaultAssignmentResult]:
    def assign_fault(task_id: int, user_instruction: str, traj: List[Dict[str, Any]], ground_truth_actions: List[Action], ground_truth_outputs: List[str]) -> FaultAssignmentResult:
        idx_to_author = {
            0: FaultAuthor.USER,
            1: FaultAuthor.AGENT,
            2: FaultAuthor.ENVIRONMENT,
        }
        grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
        ctx_desc = context_description(grading_strategy)
        context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
        res = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the entity that is responsible for the fault. The user is responsible for the fault if they caused an action that was not grounded in the user instruction. The agent is responsible for the fault if they took an action that was not correct (or took the action with the wrong arguments). The environment is responsible for all other faults.",
            text=context,
            options=["The user", "The agent", "The environment (neither user nor agent)"],
        )
        author = idx_to_author[res]
        description = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why {author.value} is responsible for the fault in the trajectory. Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultAssignmentResult(task_id=task_id, author=author, description=description)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids = [r.task_id for r in results]
        user_instructions = [r.user_instruction for r in results]
        trajs = [r.traj for r in results]
        ground_truth_actions = [r.ground_truth_actions for r in results]
        ground_truth_outputs = [r.ground_truth_outputs for r in results]
        results = list(executor.map(assign_fault, task_ids, user_instructions, trajs, ground_truth_actions, ground_truth_outputs))
    return results


def hierarchical_error_analysis(
    api: API,
    results: List[OriginalResult],
    max_concurrency: int
) -> List[List[ErrorChain]]:
    """
    Perform hierarchical error analysis on all results
    Returns list of error chain lists (one list per trajectory)
    """
    analyzer = HierarchicalErrorAnalyzer(api)

    def analyze_single(result: OriginalResult) -> List[ErrorChain]:
        return analyzer.analyze_hierarchical(result)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        error_chain_lists = list(executor.map(analyze_single, results))

    return error_chain_lists


def extract_error_path(node: ErrorNode) -> List[str]:
    """Extract category path from root to leaf"""
    path = [node.category]
    if node.children:
        path.extend(extract_error_path(node.children[0]))
    return path


def print_hierarchical_statistics(
    hierarchical_results: List[Dict[str, Any]],
    original_results: List[OriginalResult]
) -> None:
    """Print aggregate statistics for hierarchical analysis"""

    print("\n" + "=" * 80)
    print("HIERARCHICAL ERROR ANALYSIS RESULTS")
    print("=" * 80)

    # Flatten all chains from all trajectories
    all_chains = []
    chains_per_trajectory = []
    for r in hierarchical_results:
        chains = r.get("error_chains", [])
        all_chains.extend(chains)
        chains_per_trajectory.append(len(chains))

    total_trajectories = len(hierarchical_results)
    total_chains = len(all_chains)

    print(f"\nTotal Trajectories: {total_trajectories}")
    print(f"Total Error Chains: {total_chains}")
    print(f"Average Chains per Trajectory: {total_chains / total_trajectories:.2f}")

    # Priority distribution
    priorities = Counter([chain["priority"] for chain in all_chains])
    print("\nError Priority Distribution:")
    for priority in ["critical", "major", "minor"]:
        count = priorities.get(priority, 0)
        percentage = count / total_chains * 100 if total_chains > 0 else 0
        icon = {"critical": "🔥", "major": "⚠️", "minor": "ℹ️"}.get(priority, "")
        print(f"  {icon} {priority}: {count} ({percentage:.1f}%)")

    # Root cause distribution
    root_causes = Counter([chain["root_cause"] for chain in all_chains])
    print("\nRoot Cause Distribution:")
    for cause, count in root_causes.most_common():
        percentage = count / total_chains * 100 if total_chains > 0 else 0
        print(f"  - {cause}: {count} ({percentage:.1f}%)")

    # High-level error distribution (from chains)
    high_level_errors = Counter([
        chain["chain"]["children"][0]["category"]
        for chain in all_chains
        if chain["chain"].get("children")
    ])
    print("\nHigh-Level Error Type Distribution:")
    for error_type, count in high_level_errors.most_common():
        percentage = count / total_chains * 100 if total_chains > 0 else 0
        print(f"  - {error_type}: {count} ({percentage:.1f}%)")

    # Mid-level error distribution
    mid_level_errors = Counter([
        chain["chain"]["children"][0]["children"][0]["category"]
        for chain in all_chains
        if (chain["chain"].get("children") and
            chain["chain"]["children"][0].get("children"))
    ])
    print("\nMid-Level Error Manifestation Distribution:")
    for error_type, count in mid_level_errors.most_common():
        percentage = count / total_chains * 100 if total_chains > 0 else 0
        print(f"  - {error_type}: {count} ({percentage:.1f}%)")

    # Low-level error distribution
    low_level_errors = Counter([
        chain["chain"]["children"][0]["children"][0]["children"][0]["category"]
        for chain in all_chains
        if (chain["chain"].get("children") and
            chain["chain"]["children"][0].get("children") and
            chain["chain"]["children"][0]["children"][0].get("children"))
    ])
    print("\nLow-Level Technical Error Distribution:")
    for error_type, count in low_level_errors.most_common():
        percentage = count / total_chains * 100 if total_chains > 0 else 0
        print(f"  - {error_type}: {count} ({percentage:.1f}%)")

    # Most common error paths
    error_paths = []
    for chain in all_chains:
        try:
            path = []
            current = chain["chain"]
            while current:
                path.append(current["category"])
                current = current.get("children", [None])[0]
            error_paths.append(" → ".join(path))
        except (KeyError, IndexError, TypeError):
            continue

    print("\nMost Common Error Paths (Root → High → Mid → Low):")
    for path, count in Counter(error_paths).most_common(10):
        print(f"  {count}x: {path}")

    print("\n" + "=" * 80)


def fault_type_analysis(api: API, results: List[OriginalResult], max_concurrency: int) -> List[FaultTypeResult]:
    def get_fault_type(task_id: int, user_instruction: str, traj: List[Dict[str, Any]], ground_truth_actions: List[Action], ground_truth_outputs: List[str]) -> FaultTypeResult:
        idx_to_fault_type = {
            0: FaultType.CALLED_WRONG_TOOL,
            1: FaultType.USED_WRONG_TOOL_ARGUMENT,
            2: FaultType.GOAL_PARTIALLY_COMPLETED,
            3: FaultType.OTHER,
        }
        grading_strategy = GradingStrategy.OUTPUTS if len(ground_truth_outputs) > 0 else GradingStrategy.ACTIONS
        ctx_desc = context_description(grading_strategy)
        context = display_context(user_instruction, ground_truth_actions, ground_truth_outputs, traj)
        res = api.classify(
            instruction=f"{ctx_desc}\n\nDetermine the type of fault of the first instance of the fault.",
            text=context,
            options=["The user called the wrong tool", "The user used the correct tool with a wrong argument", "The goal was only partially completed", "Other"],
        )
        fault_type = idx_to_fault_type[res]
        description = api.generate(
            instruction=f"{ctx_desc}\n\nDescribe the reason why the following trajectory contains a fault of type \"{fault_type.value}\". Be concise and only focus on the functional differences between the ground truth and the trajectory.",
            text=context,
        )
        return FaultTypeResult(task_id=task_id, fault_type=fault_type, description=description)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        task_ids = [r.task_id for r in results]
        user_instructions = [r.user_instruction for r in results]
        trajs = [r.traj for r in results]
        ground_truth_actions = [r.ground_truth_actions for r in results]
        ground_truth_outputs = [r.ground_truth_outputs for r in results]
        results = list(executor.map(get_fault_type, task_ids, user_instructions, trajs, ground_truth_actions, ground_truth_outputs))
    return results

def main() -> None:
    args = get_args()
    api = default_api_from_args(args)
    with open(args.results_path, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results")
    # Load tasks from benchmark if env is specified
    tasks = None
    if args.env:
        env = args.env
        if env == "airline":
            tasks: List[Task] = AIRLINE_TASKS
        elif env == "retail":
            tasks: List[Task] = RETAIL_TASKS
        else:
            raise ValueError(f"Invalid environment: {env}")
    failed_results = [r for r in results if r["reward"] <= 1e-3]
    print(f"Found {len(failed_results)} failed trajectories")
    if args.max_num_failed_results is not None and len(failed_results) > args.max_num_failed_results:
        print(f"Limiting to {args.max_num_failed_results} failed trajectories")
        failed_results = failed_results[:args.max_num_failed_results]
    original_results = []
    for result in failed_results:
        task_id: int = result["task_id"]

        # handle examples where there's "error" key instead of "task", pass those through
        if "info" in result and "error" in result["info"]:
            continue

        # Check if task info is embedded in result (new format) or needs to be fetched from tasks list
        if "info" in result and "task" in result["info"]:
            # New format: task info embedded in result
            task_info = result["info"]["task"]
            user_instruction = task_info["instruction"]
            ground_truth_outputs = task_info.get("outputs", [])

            # Convert actions from dict format to Action objects
            ground_truth_actions = []
            for action_dict in task_info.get("actions", []):
                action = Action(
                    name=action_dict["name"],
                    kwargs=action_dict.get("kwargs", action_dict.get("arguments", {}))
                )
                ground_truth_actions.append(action)
        else:
            # Old format: fetch from tasks list
            task = tasks[task_id]
            user_instruction = task.instruction
            ground_truth_actions = task.actions
            ground_truth_outputs = task.outputs

        original_result = OriginalResult(
            task_id=task_id,
            user_instruction=user_instruction,
            traj=result["traj"],
            ground_truth_actions=ground_truth_actions,
            ground_truth_outputs=ground_truth_outputs
        )
        original_results.append(original_result)
    # Choose analysis mode based on --hierarchical flag
    if args.hierarchical:
        print(f"\nPerforming HIERARCHICAL error analysis on {len(original_results)} failed trajectories...")
        print(f"This will identify ALL distinct errors and analyze each at 4 levels")
        print(f"Max concurrency: {args.max_concurrency}\n")

        error_chain_lists = hierarchical_error_analysis(
            api=api,
            results=original_results,
            max_concurrency=args.max_concurrency
        )

        # Print detailed chains for first few examples
        print("\n" + "=" * 80)
        print("SAMPLE ERROR CHAINS (showing first 3 trajectories)")
        print("=" * 80)
        analyzer = HierarchicalErrorAnalyzer(api)
        for i, (result, chains) in enumerate(zip(original_results[:3], error_chain_lists[:3])):
            print(f"\n{'='*80}")
            print(f"TASK {result.task_id}: Found {len(chains)} error chain(s)")
            print(f"{'='*80}")
            print(analyzer.visualize_chains(chains))

        # Prepare results for saving
        hierarchical_results = []
        for result, chains in zip(original_results, error_chain_lists):
            hierarchical_results.append({
                "task_id": result.task_id,
                "num_errors": len(chains),
                "error_chains": [chain.model_dump() for chain in chains]
            })

        # Print aggregate statistics
        print_hierarchical_statistics(hierarchical_results, original_results)

        # Save results
        with open(args.output_path, "w") as f:
            json.dump({
                "analysis_mode": "hierarchical",
                "hierarchical_error_analysis": hierarchical_results,
            }, f, indent=4)
        print(f"\nSaved hierarchical analysis results to {args.output_path}")

    else:
        # Original flat analysis mode
        print(f"Performing fault assignment analysis on {len(original_results)} failed trajectories with a max concurrency of {args.max_concurrency}...")
        fault_assignment_results = fault_assignment_analysis(api=api, results=original_results, max_concurrency=args.max_concurrency)
        failures_due_to_agent = [original_results[i] for i, r in enumerate(fault_assignment_results) if r.author == FaultAuthor.AGENT]
        print(f"Performing fault type analysis on {len(failures_due_to_agent)} failures that have been marked as being caused by the agent with a max concurrency of {args.max_concurrency}...")
        fault_type_results = fault_type_analysis(api=api, results=failures_due_to_agent, max_concurrency=args.max_concurrency)
        print(f"""Reviewed {len(fault_assignment_results)} trajectories:

Author fault distribution:
  - User: {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.USER)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.USER) / len(fault_assignment_results) * 100, 2)}%)
  - Agent: {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.AGENT)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.AGENT) / len(fault_assignment_results) * 100, 2)}%)
  - Environment (otherwise case): {sum(1 for r in fault_assignment_results if r.author == FaultAuthor.ENVIRONMENT)} ({round(sum(1 for r in fault_assignment_results if r.author == FaultAuthor.ENVIRONMENT) / len(fault_assignment_results) * 100, 2)}%)

Fault type distribution (only failures marked as being caused by the agent):
  - Called wrong tool: {sum(1 for r in fault_type_results if r.fault_type == FaultType.CALLED_WRONG_TOOL)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.CALLED_WRONG_TOOL) / len(fault_type_results) * 100, 2)}%)
  - Used wrong tool argument: {sum(1 for r in fault_type_results if r.fault_type == FaultType.USED_WRONG_TOOL_ARGUMENT)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.USED_WRONG_TOOL_ARGUMENT) / len(fault_type_results) * 100, 2)}%)
  - Goal partially completed: {sum(1 for r in fault_type_results if r.fault_type == FaultType.GOAL_PARTIALLY_COMPLETED)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.GOAL_PARTIALLY_COMPLETED) / len(fault_type_results) * 100, 2)}%)
  - Other: {sum(1 for r in fault_type_results if r.fault_type == FaultType.OTHER)} ({round(sum(1 for r in fault_type_results if r.fault_type == FaultType.OTHER) / len(fault_type_results) * 100, 2)}%)
""")
        with open(args.output_path, "w") as f:
            json.dump({
                "analysis_mode": "flat",
                "fault_assignment_analysis": [r.model_dump() for r in fault_assignment_results],
                "fault_type_analysis": [r.model_dump() for r in fault_type_results],
            }, f, indent=4)
        print(f"Saved results to {args.output_path}")

if __name__ == "__main__":
    main()
