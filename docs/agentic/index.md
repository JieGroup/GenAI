# 11. Agentic AI

## Introduction to Agents and Agentic AI Workflows

According to Google’s introduction, an AI agent is essentially a system that brings together a model, a set of tools, an orchestration mechanism, and the runtime components required to let the model operate in a loop toward a goal. 
These elements collectively provide the foundational structure of any autonomous system.

### Core Components of an AI Agent

- **A Model (the “Brain”)**  
  The core language model or foundation model that performs reasoning, evaluates options, and makes decisions. The model type (general-purpose, fine-tuned, or multimodal) determines the agent’s cognitive capability.

- **Tools (the “Hands”)**  
  Interfaces that connect the agent’s reasoning to the external world. These include API calls, code functions, databases, vector stores, or browser tools.Tools enable the agent to gather information and execute real-world actions.

- **An Orchestration Layer (the “Nervous System”)**  
  The governing process that manages planning, memory, and reasoning. This layer decides when to think, when to act, and how to break tasks into steps using methods such as Chain-of-Thought or ReAct.

- **Deployment Infrastructure (the “Body & Legs”)**  
  The environment that hosts the agent as a reliable, secure, and scalable service. It provides monitoring, logging, and interfaces such as Agent-to-Agent (A2A) APIs.

---

## What Is Agentic AI?

Agentic AI is not a system, but a way of working — a workflow where the model operates through multiple steps, makes autonomous decisions, invokes tools when needed, performs self-correction, keeps track of state, engages in long-horizon reasoning, and completes tasks step by step.


### Example

An “AI travel-planning team” organizing a trip: one model checks flight schedules and prices; one finds hotel options and compares amenities; one builds a daily itinerary based on user preferences; one reviews the entire plan, detects conflicts, and resolves issues.
You only specify the destination and travel dates; the agent figures out where to search, how to compare options, and how to optimize the final plan. The workflow emphasizes multi-step planning, tool usage (search, APIs, calendars), and self-refinement, rather than simply generating text.

## Agentic Design Patterns

Following Andrew Ng’s framework, the behaviors of an agentic system can be grouped into four major design patterns.  
These patterns describe **how an AI agent operates**, plans, corrects itself, and collaborates to solve complex tasks.

**Figure: Agentic Design Patterns** (Source: Online image (copyright holder not identifiable). Included under educational fair use.)
<div style="text-align:center;">
    <img src="../_static/img/agentic patterns.png" alt="Sample Aug" width="600" style="display:block; margin:auto;">
</div>
---

### Reflection — Self-Evaluation and Iterative Improvement

Reflection refers to an agent’s ability to **inspect its own output**, identify weaknesses, and produce a refined version.

Typical workflow:

1. **Initial attempt**:  
   The model generates a first-pass result (e.g., a draft, code snippet, summary).

2. **Self-critique or external critique**:  
   The initial result is passed back into the same model—or a separate “reviewer” model—for evaluation.  
   The reviewer checks for correctness, style, efficiency, clarity, or logical consistency.

3. **Revision**:  
   The agent produces an improved version based on the critique (e.g., fixing bugs, restructuring reasoning, tightening arguments).

4. **Iteration**:  
   This loop continues until the output reaches the desired quality level.

**Key ideas**

- The reviewer can be the same model (“self-reflection”) or a dedicated evaluation model.
- Reflection criteria may be objective (e.g., “does this code run?”) or subjective (e.g., writing style, clarity).

---

### Tool Use — Extending the Agent Beyond Language

Tool use enables an agent to interact with the outside world rather than staying confined to text generation.

Common tool families:

- **Analysis**: code execution, math solvers, Wolfram Alpha  
- **Information acquisition**: web search, Wikipedia lookup, database queries  
- **Productivity tools**: email, calendar, messaging  
- **Image tools**: OCR, captioning, generation  

General workflow:

1. **Determine the need**:  
   The agent recognizes that a task requires an external capability (e.g., searching, computing, parsing).

2. **Generate tool call**:  
   The agent produces the structured arguments required by the selected tool.

3. **Execute and receive results**:  
   The external tool performs the action and returns data.

4. **Integrate results**:  
   The agent incorporates tool outputs into the broader workflow to finish the task.

**Key idea**

- Tool use dramatically expands what an agent can accomplish, enabling real-world interaction and solving tasks that pure text generation cannot handle.

**Sample code**

Below is a simple example showing how an agent can detect a need for a tool, generate a tool call, execute the tool, and integrate the result.

```python
# Minimal Agentic Workflow Example (LangChain)

import os

# 0. Set your API key before running
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


# 1. Define a simple tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

multiply_tool = Tool(
    name="Multiply",
    func=lambda x: multiply(*map(int, x.split(","))),
    description="Multiply two integers. Input format: 'a,b' (for example: '12,9').",
)


# 2. Initialize the base LLM (the agent's “brain”)
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or any chat model you use in class
    temperature=0.0,
)


# 3. Create an agent that can plan → call tools → reason → answer
agent = initialize_agent(
    tools=[multiply_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,          # prints the reasoning and tool calls — good for teaching
)


# 4. Ask the agent to solve a multi-step task
question = (
    "If Alice has 12 packs of snacks and each pack has 9 items, "
    "how many total items are there? "
    "Think step by step and use the Multiply tool when you need to."
)

response = agent.run(question)
print("Final answer:", response)

```
---

### Planning — Decomposing Complex Tasks Into Actionable Steps

Planning captures an agent’s ability to **analyze a complicated request** and **design a step-by-step strategy** to fulfill it—without a developer hard-coding the logic.

General process:

1. **Task comprehension**:  
   The agent interprets a high-level instruction (e.g., “Generate a new image matching the pose of the person in the example”).

2. **Subtask decomposition**:  
   The system breaks the request into smaller operations and determines which tools/models are required.

3. **Execution pipeline**:  
   The agent organizes these operations into a sequence (a mini-workflow). 

**Key ideas**

- Planning increases flexibility but is more challenging to control.
- When facing unseen tasks, the agent can creatively assemble tools into novel workflows.

---

### Multi-Agent Collaboration — Distributed Roles and Cooperative Problem Solving

Instead of relying on a single model, multiple agents can work together, each specializing in a particular role.

Collaboration framework:

1. **Role assignment**:  
Each agent is given a purpose (e.g., researcher, analyst, editor, critic).

2. **Division of labor**:  
Agents tackle different parts of the overall task using their strengths.

3. **Communication loops**:  
Agents exchange messages, share intermediate outputs, negotiate decisions, and align on results.

**Key ideas**

- Multi-agent systems often outperform single agents on complex tasks such as writing biographies, analyzing documents, or playing strategy games.
- Research shows measurable improvements when tasks benefit from diverse perspectives.
- Drawback: harder to control and predict, since multiple autonomous entities interact.

---

## Is Multi-Agent More Effective Than Single-Agent?

The adoption of multi-agent systems (MAS) has been widely explored across AI, robotics, distributed systems, and decision-making applications. Proposed advantages often include greater computational efficiency, improved adaptability, and enhanced robustness. However, whether MAS are inherently more capable than single-agent systems remains an open question.

To reason about this question more systematically, we consider three connected analytical perspectives:
- **Task Allocation**:  Task allocation considers how MAS can dynamically divide
and coordinate tasks more effectively than single agents, drawing from divide-and-conquer strategies. 
- **Robustness Under Uncertainty**: Robustness examines whether these benefits hold under uncertainty and failure, relating closely to ensemble learning principles 
- **Feedback Integration and Adaptation**: Feedback integration explores how MAS adapt over time through
internal and external signals, leveraging Bayesian approaches.


## Safety in Multi-Agent AI Systems

Unlike single-agent architectures, where failures tend to remain localized, MAS exhibit complex interdependencies that can propagate or even amplify existing vulnerabilities. These interdependencies significantly complicate efforts to ensure robustness, reliability, and security across the entire system.

A critical but often overlooked factor in MAS safety is the topology—the way agents are connected and how information flows between them. Different interaction structures can either amplify a backdoor signal or dampen it as it travels through the system. Here we summarize two canonical topologies frequently studied in the literature:

- Star topology — Backdoor amplification depends on agent similarity: In a star topology, if agents share similar vulnerabilities, their correlated outputs amplify the backdoor effect through aggregation; if vulnerabilities are diverse, aggregation can attenuate the attack.
- Cascade topology — Early mistakes propagate and grow: In a cascade topology, early-stage backdoor signals can be magnified as they flow downstream, causing severe degradation; but if intermediate agents disrupt or filter the malicious signal, the cascade can dilute or break the backdoor effect.





