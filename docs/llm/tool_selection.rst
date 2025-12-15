====================
Tool Selection
====================

Overview
========

Tool selection is a critical capability for coding agents, enabling them to choose and use appropriate tools from a toolkit to accomplish tasks. This section covers strategies, frameworks, and best practices for tool selection in LLM-based agents.

What are Tools?
===============

Definition
----------

Tools are external capabilities that extend an LLM's abilities beyond text generation:

* **Code Execution:** Running Python, JavaScript, shell commands
* **File Operations:** Reading, writing, editing files
* **Search:** Web search, code search, documentation lookup
* **APIs:** Calling external services
* **Database Operations:** Querying and modifying data
* **Testing:** Running tests, linting, formatting

Types of Tools
--------------

Deterministic Tools
~~~~~~~~~~~~~~~~~~~

* File system operations
* Code formatters
* Compilers
* Static analysis tools

Non-Deterministic Tools
~~~~~~~~~~~~~~~~~~~~~~~

* Web search
* API calls with variable responses
* User input requests

Stateful Tools
~~~~~~~~~~~~~~

* Database connections
* REPL environments
* Interactive debuggers

Tool Selection Challenges
=========================

Choice Complexity
-----------------

* Large tool inventories
* Overlapping capabilities
* Context-dependent suitability

Cost Considerations
-------------------

* Execution time
* API costs
* Resource consumption

Error Handling
--------------

* Tool failures
* Malformed inputs
* Unexpected outputs

Tool Selection Strategies
=========================

Direct Selection
----------------

LLM directly chooses tools based on task description.

**Approach:**

1. Provide tool descriptions in prompt
2. LLM selects tool(s) and parameters
3. Execute selected tools
4. Continue based on results

**Pros:**

* Simple implementation
* Flexible
* Works with any LLM

**Cons:**

* Depends entirely on LLM capability
* No optimization
* May make suboptimal choices

ReAct (Reasoning + Acting)
---------------------------

Interleaves reasoning and tool use.

**Pattern:**

.. code-block:: text

    Thought: [Reasoning about what to do]
    Action: [Tool name]
    Action Input: [Tool parameters]
    Observation: [Tool output]
    Thought: [Reasoning about observation]
    ...

**Benefits:**

* Explicit reasoning traces
* Better explainability
* Handles multi-step tasks

**Example:**

.. code-block:: text

    Thought: I need to find the definition of the calculate_total function
    Action: code_search
    Action Input: "def calculate_total"
    Observation: Found in src/billing/calculator.py:42
    Thought: Now I should read that file to understand the implementation
    Action: read_file
    Action Input: src/billing/calculator.py
    ...

Function Calling / Tool Use APIs
---------------------------------

Structured tool invocation via API formats.

**Formats:**

* OpenAI Function Calling
* Anthropic Tool Use
* Custom schemas

**Advantages:**

* Structured outputs
* Better parsing
* Reduced errors

**Example Schema:**

.. code-block:: json

    {
      "name": "search_code",
      "description": "Search for code patterns in the repository",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {"type": "string", "description": "Regex pattern to search"},
          "file_type": {"type": "string", "description": "File extension filter"}
        },
        "required": ["pattern"]
      }
    }

Planning-Based Selection
------------------------

Plan tool sequence before execution.

**Process:**

1. Analyze task requirements
2. Create tool usage plan
3. Validate plan feasibility
4. Execute plan with monitoring

**Benefits:**

* Optimized tool sequences
* Better resource allocation
* Easier debugging

Learning-Based Selection
------------------------

Learn optimal tool selection from data.

**Approaches:**

* Reinforcement learning
* Supervised learning from demonstrations
* Few-shot learning with examples

Tool Selection Patterns
=======================

Sequential Tool Use
-------------------

Tools used one after another, each building on previous results.

**Example:**

1. Search for relevant code
2. Read found files
3. Analyze code
4. Generate fix
5. Run tests

Parallel Tool Use
-----------------

Multiple tools executed simultaneously.

**Use Cases:**

* Gathering information from multiple sources
* Running independent checks
* Parallel test execution

Conditional Tool Use
--------------------

Tool selection based on runtime conditions.

**Pattern:**

.. code-block:: text

    IF error_type == "syntax":
        USE linter_tool
    ELIF error_type == "runtime":
        USE debugger_tool
    ELSE:
        USE code_search

Iterative Tool Use
------------------

Repeated tool application until goal achieved.

**Example:**

.. code-block:: text

    WHILE tests_failing:
        1. Analyze failures
        2. Generate fix
        3. Run tests
        4. IF all_pass: BREAK

Tool Composition
----------------

Chaining tools to create complex capabilities.

**Example:**

.. code-block:: text

    code_search -> read_file -> analyze_code -> suggest_fix -> apply_fix -> run_tests

Tool Description Design
=======================

Effective Descriptions
----------------------

**Components:**

* Clear name
* Concise description
* Parameter specifications
* Usage examples
* Constraints and limitations

**Example:**

.. code-block:: yaml

    name: grep_code
    description: |
      Search for text patterns in code files using regex.
      Best for finding specific strings, function names, or patterns.
    parameters:
      - name: pattern
        type: string
        required: true
        description: Regular expression to search for
      - name: path
        type: string
        required: false
        description: Directory or file path to search in (default: current directory)
      - name: file_extension
        type: string
        required: false
        description: Filter by file extension (e.g., "py", "js")
    examples:
      - Find all TODO comments: grep_code "TODO:" --path src/
      - Find function definitions: grep_code "def \w+\(" --file_extension py
    limitations:
      - Maximum file size: 1MB
      - Binary files are skipped

Tool Discovery
==============

Static Tool Lists
-----------------

Predefined set of available tools.

**Pros:**

* Simple
* Predictable
* Easy to optimize prompts

**Cons:**

* Not extensible at runtime
* Fixed capability set

Dynamic Tool Discovery
----------------------

Tools discovered or registered at runtime.

**Mechanisms:**

* Tool registries
* Plugin systems
* Service discovery

Context-Aware Tools
-------------------

Tool availability based on context.

**Examples:**

* Project-specific tools
* Language-specific tools
* Environment-dependent tools

Tool Execution
==============

Sandboxing
----------

* Isolated execution environments
* Resource limits
* Permission controls
* Timeout mechanisms

Error Handling
--------------

**Strategies:**

* Retry with corrections
* Fallback to alternative tools
* Request clarification
* Graceful degradation

Result Validation
-----------------

* Type checking
* Schema validation
* Semantic verification
* Side-effect confirmation

Multi-Tool Coordination
=======================

Tool Dependencies
-----------------

Managing prerequisites and sequencing.

**Example:**

* Must read file before editing
* Must compile before running
* Must install dependencies before testing

Conflict Resolution
-------------------

Handling contradictory tool actions.

**Strategies:**

* Priority-based resolution
* User confirmation
* Rollback mechanisms

Resource Management
-------------------

* Token budget allocation
* API rate limiting
* Execution time limits
* Memory constraints

Tool Selection Optimization
============================

Prompt Engineering
------------------

**Techniques:**

* Clear tool descriptions
* Usage examples
* Decision criteria
* Negative examples (when NOT to use)

Tool Grouping
-------------

Organizing tools by category or capability.

**Categories:**

* File operations
* Code analysis
* Execution & testing
* Search & discovery
* External services

Tool Ranking
------------

Prioritizing tools based on:

* Success rate
* Execution cost
* Latency
* Relevance to task

Frameworks & Protocols
======================

LangChain Tools
---------------

* Tool abstraction
* Agent executors
* Tool chains

LlamaIndex Tools
----------------

* Query engines as tools
* Data connectors
* Tool specifications

OpenAI Function Calling
-----------------------

* JSON schema definitions
* Structured outputs
* Parallel function calls

Anthropic Tool Use
------------------

* Tool schemas
* Multi-step tool use
* Tool result handling

Model Context Protocol (MCP)
-----------------------------

* Standardized tool interfaces
* Cross-platform compatibility
* See :doc:`../tools/mcp`

Evaluation & Metrics
====================

Tool Selection Accuracy
-----------------------

* Correct tool chosen for task
* Optimal vs. suboptimal selection
* Unnecessary tool invocations

Tool Use Efficiency
-------------------

* Number of tool calls
* Execution time
* Cost per task
* Success rate

Task Completion
---------------

* End-to-end success rate
* Tool-assisted vs. baseline
* Quality of results

Common Pitfalls
===============

Tool Overuse
------------

Using tools when unnecessary.

**Mitigation:**

* Clear guidelines on when tools are needed
* Cost-benefit analysis
* Encourage direct generation when appropriate

Tool Misuse
-----------

Using tools incorrectly or for wrong purposes.

**Prevention:**

* Detailed tool documentation
* Input validation
* Usage examples

Tool Hallucination
------------------

Referencing non-existent tools or capabilities.

**Solutions:**

* Strict tool schema validation
* Error messages for unknown tools
* Regular model updates

Infinite Loops
--------------

Repeatedly calling tools without progress.

**Safeguards:**

* Maximum iteration limits
* Progress tracking
* Break conditions

Best Practices
==============

For Tool Designers
------------------

1. Write clear, unambiguous descriptions
2. Provide comprehensive examples
3. Define clear input/output schemas
4. Include error conditions
5. Document limitations

For Agent Developers
--------------------

1. Start with essential tools
2. Test tool selection extensively
3. Implement robust error handling
4. Monitor tool usage patterns
5. Optimize based on metrics

For Prompt Engineers
--------------------

1. Include tool selection reasoning
2. Provide decision frameworks
3. Use examples to guide selection
4. Specify when NOT to use tools
5. Encourage explanation of choices

Research Directions
===================

* Automated tool description generation
* Learned tool selection policies
* Tool synthesis and composition
* Cross-domain tool transfer
* Adaptive tool creation

Resources
=========

Frameworks
----------

* LangChain
* LlamaIndex
* AutoGPT
* Custom implementations

Papers
------

[Add relevant papers on tool use]

Examples
--------

[Add example implementations]

See Also
========

* :doc:`reasoning`
* :doc:`planning`
* :doc:`multi_turn`
* :doc:`../tools/mcp`
* :doc:`../tools/frameworks`
