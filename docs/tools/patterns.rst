========================
Agent Design Patterns
========================

Overview
========

Design patterns provide proven solutions to common problems in agent development. This section covers architectural patterns, interaction patterns, and implementation strategies for coding agents.

Core Agent Patterns
===================

ReAct (Reason + Act)
--------------------

Interleave reasoning and action execution.

**Structure:**

.. code-block:: text

    Thought: [Reasoning step]
    Action: [Tool/action name]
    Action Input: [Parameters]
    Observation: [Result]
    ... (repeat)
    Thought: [Final reasoning]
    Answer: [Final response]

**Benefits:**

* Explicit reasoning traces
* Better explainability
* Error recovery through re-reasoning
* Handles multi-step tasks

**Implementation:**

.. code-block:: python

    class ReActAgent:
        def __init__(self, llm, tools):
            self.llm = llm
            self.tools = tools
            self.max_iterations = 10

        async def run(self, query: str) -> str:
            context = [f"Question: {query}"]

            for i in range(self.max_iterations):
                # Generate thought and action
                response = await self.llm.generate(
                    "\n".join(context) + "\nThought:"
                )

                if "Answer:" in response:
                    return self.extract_answer(response)

                thought, action, action_input = self.parse_response(response)
                context.append(f"Thought: {thought}")
                context.append(f"Action: {action}")
                context.append(f"Action Input: {action_input}")

                # Execute action
                observation = await self.execute_tool(action, action_input)
                context.append(f"Observation: {observation}")

            return "Max iterations reached"

**Use Cases:**

* Complex problem-solving
* Multi-tool workflows
* Exploratory tasks
* Debugging

ReWOO (Reasoning WithOut Observation)
--------------------------------------

Plan actions upfront, then execute without intermediate reasoning.

**Phases:**

1. **Planning:** Generate complete action plan
2. **Execution:** Run all actions
3. **Reasoning:** Process all results together

**Advantages:**

* Fewer LLM calls
* Parallelizable actions
* Lower latency
* Cost-effective

**Trade-offs:**

* Less adaptive
* Cannot handle dynamic conditions
* Requires good upfront understanding

**Example:**

.. code-block:: text

    Plan:
    1. #E1 = search_codebase("authentication")
    2. #E2 = read_file(#E1)
    3. #E3 = analyze_security(#E2)

    Execute all, then reason about results

Plan-and-Execute
-----------------

Separate planning and execution phases.

**Process:**

1. **Planning Phase:**

   * Analyze task
   * Create detailed plan
   * Identify required resources

2. **Execution Phase:**

   * Execute plan steps
   * Monitor progress
   * Adjust if needed

**Benefits:**

* Clear separation of concerns
* Easier to validate plans
* Better resource estimation
* User can review plans

**Implementation Pattern:**

.. code-block:: python

    class PlanAndExecuteAgent:
        async def run(self, task: str):
            # Planning phase
            plan = await self.create_plan(task)

            # Optional: User approval
            if not await self.approve_plan(plan):
                return

            # Execution phase
            results = []
            for step in plan.steps:
                result = await self.execute_step(step)
                results.append(result)

                # Check if replanning needed
                if result.needs_replan:
                    plan = await self.replan(task, results)

            return self.synthesize_results(results)

Reflexion
---------

Agent reflects on its actions and learns from mistakes.

**Loop:**

1. **Act:** Attempt task
2. **Evaluate:** Assess result
3. **Reflect:** Analyze what went wrong
4. **Refine:** Improve approach
5. **Retry:** Attempt again with insights

**Self-Reflection Prompt:**

.. code-block:: text

    You attempted to solve the task but failed.
    Previous attempts:
    - Attempt 1: [description] - Failed because [reason]
    - Attempt 2: [description] - Failed because [reason]

    Reflect on what went wrong and how to improve.
    What should you do differently?

**Benefits:**

* Learns from failures
* Improves over time
* Handles difficult problems
* Self-correcting

Tree of Thoughts (ToT)
----------------------

Explore multiple reasoning paths in a tree structure.

**Structure:**

.. code-block:: text

    Root: Initial problem
    ├── Approach 1
    │   ├── Step 1a
    │   └── Step 1b
    ├── Approach 2
    │   ├── Step 2a
    │   └── Step 2b
    └── Approach 3

**Search Strategies:**

* **BFS:** Breadth-first (explore all options at each level)
* **DFS:** Depth-first (explore one path fully first)
* **Best-first:** Prioritize promising paths

**Evaluation:**

Each node is scored for promise/viability.

**Use Cases:**

* Creative problem-solving
* Multiple solution approaches
* Complex algorithm design
* Uncertain requirements

Tool Use Patterns
=================

Tool Chaining
-------------

Sequence tools where output of one feeds into next.

**Example:**

.. code-block:: text

    search_files("*.py")
    → read_file(result)
    → analyze_code(content)
    → suggest_improvements(analysis)

**Implementation:**

.. code-block:: python

    class ToolChain:
        def __init__(self, tools: List[Tool]):
            self.tools = tools

        async def execute(self, initial_input):
            result = initial_input
            for tool in self.tools:
                result = await tool.execute(result)
            return result

Tool Branching
--------------

Conditional tool selection based on context.

**Pattern:**

.. code-block:: python

    async def conditional_tools(context):
        if context.error_type == "syntax":
            return await run_linter(context.file)
        elif context.error_type == "runtime":
            return await run_debugger(context.file)
        elif context.error_type == "logic":
            return await analyze_logic(context.file)

Parallel Tool Execution
-----------------------

Run multiple tools concurrently.

**Example:**

.. code-block:: python

    async def parallel_analysis(file_path):
        results = await asyncio.gather(
            check_style(file_path),
            check_security(file_path),
            check_performance(file_path),
            check_tests(file_path)
        )
        return aggregate_results(results)

Tool Fallback
-------------

Try alternatives if primary tool fails.

**Pattern:**

.. code-block:: python

    async def search_with_fallback(query):
        try:
            return await semantic_search(query)
        except SemanticSearchError:
            try:
                return await keyword_search(query)
            except KeywordSearchError:
                return await fuzzy_search(query)

Memory Patterns
===============

Short-Term Memory
-----------------

Recent context within conversation.

**Implementation:**

.. code-block:: python

    class ShortTermMemory:
        def __init__(self, max_messages=20):
            self.messages = deque(maxlen=max_messages)

        def add(self, message):
            self.messages.append(message)

        def get_context(self):
            return list(self.messages)

Long-Term Memory
----------------

Persistent knowledge across sessions.

**Storage Options:**

* Vector databases (Pinecone, Weaviate)
* Graph databases (Neo4j)
* Document stores (MongoDB)

**Retrieval:**

.. code-block:: python

    class LongTermMemory:
        def __init__(self, vector_db):
            self.db = vector_db

        async def store(self, content, metadata):
            embedding = await self.embed(content)
            await self.db.upsert(embedding, content, metadata)

        async def retrieve(self, query, limit=5):
            query_embedding = await self.embed(query)
            return await self.db.search(query_embedding, limit)

Working Memory
--------------

Active task context and state.

**Components:**

* Current goals
* Active variables
* Intermediate results
* Task state

**Example:**

.. code-block:: python

    class WorkingMemory:
        def __init__(self):
            self.goal_stack = []
            self.variables = {}
            self.results = []
            self.state = {}

        def push_goal(self, goal):
            self.goal_stack.append(goal)

        def pop_goal(self):
            return self.goal_stack.pop()

        def current_goal(self):
            return self.goal_stack[-1] if self.goal_stack else None

Episodic Memory
---------------

Memory of past experiences and episodes.

**Structure:**

.. code-block:: python

    @dataclass
    class Episode:
        task: str
        actions: List[Action]
        outcome: str
        success: bool
        timestamp: datetime
        lessons_learned: List[str]

    class EpisodicMemory:
        def store_episode(self, episode: Episode):
            # Store for future reference

        def recall_similar(self, current_task: str) -> List[Episode]:
            # Find similar past experiences

Error Handling Patterns
========================

Retry with Exponential Backoff
-------------------------------

Retry failed operations with increasing delays.

**Implementation:**

.. code-block:: python

    async def retry_with_backoff(
        func,
        max_retries=3,
        base_delay=1,
        max_delay=60
    ):
        for attempt in range(max_retries):
            try:
                return await func()
            except RetryableError as e:
                if attempt == max_retries - 1:
                    raise
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)

Circuit Breaker
---------------

Stop calling failing services temporarily.

**States:**

* **Closed:** Normal operation
* **Open:** Failing, reject requests
* **Half-Open:** Testing recovery

**Implementation:**

.. code-block:: python

    class CircuitBreaker:
        def __init__(self, failure_threshold=5, timeout=60):
            self.failure_count = 0
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.state = "closed"
            self.last_failure_time = None

        async def call(self, func):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise CircuitOpenError()

            try:
                result = await func()
                self.on_success()
                return result
            except Exception as e:
                self.on_failure()
                raise

Graceful Degradation
--------------------

Provide reduced functionality on failures.

**Example:**

.. code-block:: python

    async def get_code_suggestions(code):
        try:
            # Try advanced AI suggestions
            return await advanced_suggestions(code)
        except ServiceUnavailable:
            try:
                # Fall back to simpler suggestions
                return await basic_suggestions(code)
            except:
                # Minimal functionality
                return default_suggestions()

Compensation Pattern
--------------------

Undo actions on failure (Saga pattern).

**Example:**

.. code-block:: python

    class Transaction:
        def __init__(self):
            self.actions = []
            self.compensations = []

        async def execute(self):
            try:
                for action in self.actions:
                    await action()
            except Exception:
                # Rollback
                for compensation in reversed(self.compensations):
                    await compensation()
                raise

Orchestration Patterns
======================

Hierarchical Orchestration
--------------------------

Tree of agents with managers and workers.

**Structure:**

.. code-block:: python

    class OrchestratorAgent:
        def __init__(self, sub_agents):
            self.sub_agents = sub_agents

        async def delegate_task(self, task):
            # Decompose task
            subtasks = self.decompose(task)

            # Assign to sub-agents
            results = await asyncio.gather(*[
                agent.execute(subtask)
                for agent, subtask in zip(self.sub_agents, subtasks)
            ])

            # Synthesize results
            return self.synthesize(results)

Pipeline Pattern
----------------

Sequential processing stages.

**Example:**

.. code-block:: text

    Input → Stage1 → Stage2 → Stage3 → Output

**Implementation:**

.. code-block:: python

    class Pipeline:
        def __init__(self, stages):
            self.stages = stages

        async def process(self, input_data):
            data = input_data
            for stage in self.stages:
                data = await stage.process(data)
            return data

Map-Reduce Pattern
------------------

Parallel processing with aggregation.

**Phases:**

1. **Map:** Distribute work to agents
2. **Process:** Each agent processes independently
3. **Reduce:** Aggregate results

**Example:**

.. code-block:: python

    async def map_reduce(items, map_func, reduce_func):
        # Map phase
        mapped = await asyncio.gather(*[
            map_func(item) for item in items
        ])

        # Reduce phase
        return reduce_func(mapped)

    # Use case: Analyze multiple files
    results = await map_reduce(
        files,
        map_func=analyze_file,
        reduce_func=aggregate_analysis
    )

Event-Driven Pattern
--------------------

Agents react to events.

**Structure:**

.. code-block:: python

    class EventBus:
        def __init__(self):
            self.handlers = defaultdict(list)

        def subscribe(self, event_type, handler):
            self.handlers[event_type].append(handler)

        async def publish(self, event):
            for handler in self.handlers[event.type]:
                await handler(event)

    # Usage
    bus = EventBus()
    bus.subscribe("code_changed", auto_test_handler)
    bus.subscribe("code_changed", update_docs_handler)

    await bus.publish(CodeChangedEvent(file="main.py"))

Context Management Patterns
============================

Sliding Window
--------------

Keep recent N messages in context.

**Implementation:**

.. code-block:: python

    class SlidingWindowContext:
        def __init__(self, window_size=10):
            self.messages = deque(maxlen=window_size)

        def add(self, message):
            self.messages.append(message)

        def get_context(self):
            return list(self.messages)

Hierarchical Summarization
--------------------------

Multiple levels of summary detail.

**Levels:**

* **L1:** Full detail (recent)
* **L2:** Medium summary (older)
* **L3:** High-level summary (oldest)

**Implementation:**

.. code-block:: python

    class HierarchicalContext:
        def __init__(self):
            self.recent = []  # Full detail
            self.medium = []  # Summarized
            self.old = []     # High-level

        async def add(self, message):
            self.recent.append(message)

            if len(self.recent) > 10:
                summary = await self.summarize(self.recent[:5])
                self.medium.append(summary)
                self.recent = self.recent[5:]

            if len(self.medium) > 10:
                high_level = await self.summarize(self.medium[:5])
                self.old.append(high_level)
                self.medium = self.medium[5:]

Semantic Retrieval
------------------

Retrieve relevant context based on similarity.

**Process:**

1. Embed current query
2. Search vector database
3. Retrieve top-k similar items
4. Include in context

Code-Specific Patterns
=======================

Diff-Based Updates
------------------

Apply changes as diffs instead of full rewrites.

**Benefits:**

* Preserves surrounding code
* Reduces token usage
* Clearer change intent
* Better version control

Test-Driven Development
-----------------------

Write tests first, then implementation.

**Agent Workflow:**

1. Generate test cases
2. Run tests (should fail)
3. Generate implementation
4. Run tests (should pass)
5. Refactor if needed

Incremental Refinement
----------------------

Iteratively improve code quality.

**Phases:**

1. Working implementation
2. Add error handling
3. Improve performance
4. Enhance readability
5. Add documentation

Safety Patterns
===============

Human-in-the-Loop
-----------------

Require human approval for critical actions.

**Implementation:**

.. code-block:: python

    async def execute_with_approval(action, description):
        print(f"About to execute: {description}")
        approval = await get_user_approval()

        if approval:
            return await action()
        else:
            return "Action cancelled by user"

Sandboxing
----------

Execute code in isolated environment.

**Approaches:**

* Docker containers
* Virtual machines
* Restricted Python environments
* WebAssembly sandboxes

Rate Limiting
-------------

Prevent excessive API/tool usage.

**Implementation:**

.. code-block:: python

    class RateLimiter:
        def __init__(self, max_calls, time_window):
            self.max_calls = max_calls
            self.time_window = time_window
            self.calls = deque()

        async def acquire(self):
            now = time.time()

            # Remove old calls
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                wait_time = self.calls[0] + self.time_window - now
                await asyncio.sleep(wait_time)

            self.calls.append(time.time())

Audit Logging
-------------

Track all agent actions for accountability.

**What to Log:**

* Actions taken
* Decisions made
* Tools used
* Errors encountered
* User interactions

Best Practices
==============

Pattern Selection
-----------------

1. **Understand the Problem:** Match pattern to requirements
2. **Start Simple:** Begin with basic patterns
3. **Iterate:** Add complexity as needed
4. **Measure:** Validate pattern effectiveness
5. **Document:** Explain pattern choices

Combining Patterns
------------------

Patterns can and should be combined:

* ReAct + Reflexion: Reasoning with self-correction
* Plan-and-Execute + Human-in-Loop: Approved plans
* Tool Chaining + Circuit Breaker: Resilient pipelines

Anti-Patterns
-------------

Avoid these common mistakes:

* **Over-engineering:** Too complex for the task
* **Tight Coupling:** Hard to modify or test
* **No Error Handling:** Brittle systems
* **Ignoring Context Limits:** Token overflow
* **Sequential When Parallel:** Unnecessary latency

Resources
=========

Frameworks Implementing Patterns
---------------------------------

* LangChain
* LangGraph
* LlamaIndex
* CrewAI
* AutoGen

Papers
------

* ReAct: "ReAct: Synergizing Reasoning and Acting in Language Models"
* Reflexion: "Reflexion: Language Agents with Verbal Reinforcement Learning"
* Tree of Thoughts: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"

See Also
========

* :doc:`frameworks`
* :doc:`mcp`
* :doc:`a2a`
* :doc:`../llm/planning`
* :doc:`../llm/reasoning`
