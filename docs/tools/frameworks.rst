========================
Frameworks & Libraries
========================

Overview
========

This section covers major frameworks and libraries for building coding agents, from general-purpose agent frameworks to specialized code generation tools.

General Agent Frameworks
=========================

LangChain
---------

Comprehensive framework for building LLM applications and agents.

**Key Features:**

* Agent executors and toolkits
* Memory management
* Chain composition
* Document loaders and retrievers
* Multi-modal support

**Core Components:**

.. code-block:: python

    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.chat_models import ChatOpenAI

    # Define tools
    tools = [
        Tool(
            name="CodeSearch",
            func=search_code,
            description="Search codebase for patterns"
        )
    ]

    # Initialize agent
    llm = ChatOpenAI(temperature=0)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Run agent
    result = agent.run("Find all TODO comments")

**Agent Types:**

* Zero-shot ReAct
* Conversational ReAct
* OpenAI Functions
* Structured Chat

**Use Cases:**

* Custom coding assistants
* Document Q&A systems
* Multi-tool workflows
* RAG applications

**Resources:**

* Website: https://langchain.com
* Docs: https://python.langchain.com
* GitHub: https://github.com/langchain-ai/langchain

LangGraph
---------

Graph-based orchestration for complex agent workflows.

**Key Features:**

* State machine definition
* Conditional routing
* Cycles and loops
* Human-in-the-loop
* Streaming support

**Example:**

.. code-block:: python

    from langgraph.graph import StateGraph, END

    # Define graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_code)
    workflow.add_node("generate", generate_fix)
    workflow.add_node("test", run_tests)

    # Add edges
    workflow.add_edge("analyze", "generate")
    workflow.add_conditional_edges(
        "test",
        should_continue,
        {
            "continue": "generate",
            "end": END
        }
    )

    workflow.set_entry_point("analyze")
    app = workflow.compile()

    # Run
    result = app.invoke(initial_state)

**Advanced Features:**

* Persistence and checkpointing
* Time travel debugging
* Parallel execution
* Sub-graphs

**Use Cases:**

* Complex multi-agent systems
* Iterative refinement workflows
* Conditional logic agents
* Production-grade applications

LlamaIndex
----------

Framework for building context-augmented LLM applications.

**Key Features:**

* Data connectors
* Index structures
* Query engines
* Agent tools
* Retrieval strategies

**Data Loading:**

.. code-block:: python

    from llama_index import SimpleDirectoryReader, VectorStoreIndex

    # Load documents
    documents = SimpleDirectoryReader('./docs').load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query("How does authentication work?")

**Agent Integration:**

.. code-block:: python

    from llama_index.agent import OpenAIAgent
    from llama_index.tools import QueryEngineTool

    # Create tools from indexes
    tools = [
        QueryEngineTool(
            query_engine=docs_engine,
            metadata=ToolMetadata(
                name="docs",
                description="Search documentation"
            )
        ),
        QueryEngineTool(
            query_engine=code_engine,
            metadata=ToolMetadata(
                name="code",
                description="Search codebase"
            )
        )
    ]

    # Create agent
    agent = OpenAIAgent.from_tools(tools)
    response = agent.chat("Find authentication implementation")

**Use Cases:**

* Code documentation Q&A
* Repository understanding
* Semantic code search
* Knowledge base agents

Haystack
--------

End-to-end NLP framework with LLM support.

**Key Features:**

* Pipeline architecture
* Multiple retrieval methods
* Agent capabilities
* Production-ready
* Self-hosted option

**Example:**

.. code-block:: python

    from haystack.agents import Agent
    from haystack.tools import Tool

    # Define tools
    tools = [
        Tool(
            name="DocumentSearch",
            pipeline_or_node=document_search_pipeline,
            description="Search project documentation"
        )
    ]

    # Create agent
    agent = Agent(
        llm=llm,
        tools=tools,
        max_iterations=10
    )

    result = agent.run("How do I configure the database?")

**Use Cases:**

* Enterprise search
* Question answering
* Document processing
* Hybrid search systems

AutoGen (Microsoft)
-------------------

Multi-agent conversation framework from Microsoft Research.

**Key Features:**

* Conversable agents
* Group chat
* Human proxy agent
* Code execution
* Teaching and learning

**Example:**

.. code-block:: python

    from autogen import AssistantAgent, UserProxyAgent

    # Configure agents
    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"work_dir": "coding"}
    )

    # Start conversation
    user_proxy.initiate_chat(
        assistant,
        message="Write a function to calculate fibonacci numbers"
    )

**Multi-Agent:**

.. code-block:: python

    from autogen import GroupChat, GroupChatManager

    # Multiple specialized agents
    planner = AssistantAgent("planner", llm_config=planner_config)
    coder = AssistantAgent("coder", llm_config=coder_config)
    tester = AssistantAgent("tester", llm_config=tester_config)
    critic = AssistantAgent("critic", llm_config=critic_config)

    # Group chat
    groupchat = GroupChat(
        agents=[planner, coder, tester, critic, user_proxy],
        messages=[],
        max_round=20
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_proxy.initiate_chat(
        manager,
        message="Build a REST API for user management"
    )

**Use Cases:**

* Collaborative coding
* Code review workflows
* Teaching programming
* Complex problem-solving

CrewAI
------

Role-based multi-agent orchestration.

**Key Features:**

* Role specialization
* Task delegation
* Process automation
* Collaboration patterns

**Example:**

.. code-block:: python

    from crewai import Agent, Task, Crew

    # Define agents with roles
    researcher = Agent(
        role="Code Researcher",
        goal="Find relevant code patterns",
        backstory="Expert at code archaeology",
        tools=[code_search_tool],
        verbose=True
    )

    developer = Agent(
        role="Developer",
        goal="Implement clean, efficient code",
        backstory="Senior software engineer",
        tools=[code_generator_tool],
        verbose=True
    )

    # Define tasks
    research_task = Task(
        description="Find authentication patterns in codebase",
        agent=researcher
    )

    development_task = Task(
        description="Implement OAuth authentication",
        agent=developer
    )

    # Create crew
    crew = Crew(
        agents=[researcher, developer],
        tasks=[research_task, development_task],
        verbose=2
    )

    result = crew.kickoff()

**Use Cases:**

* Simulating development teams
* Workflow automation
* Role-based task distribution
* Collaborative projects

Semantic Kernel (Microsoft)
----------------------------

Enterprise-grade AI orchestration.

**Key Features:**

* Plugin system
* Memory and planning
* Multi-language (C#, Python, Java)
* Enterprise integration

**Example:**

.. code-block:: python

    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    # Initialize kernel
    kernel = sk.Kernel()

    # Add AI service
    kernel.add_text_completion_service(
        "openai",
        OpenAIChatCompletion("gpt-4", api_key)
    )

    # Import plugins
    code_plugin = kernel.import_skill(code_skill_instance)

    # Run function
    result = await kernel.run_async(
        code_plugin["GenerateCode"],
        input_vars={"description": "Parse JSON"}
    )

**Use Cases:**

* Enterprise applications
* Complex workflows
* Plugin ecosystems
* .NET integration

Code-Specific Frameworks
=========================

Continue.dev
------------

VS Code and JetBrains extension for AI code assistance.

**Features:**

* Tab autocomplete
* Natural language editing
* Codebase Q&A
* Slash commands
* Custom models

**Configuration:**

.. code-block:: json

    {
      "models": [{
        "title": "GPT-4",
        "provider": "openai",
        "model": "gpt-4"
      }],
      "customCommands": [
        {
          "name": "test",
          "description": "Generate unit tests",
          "prompt": "Generate comprehensive unit tests for {{{ input }}}"
        }
      ]
    }

**Use Cases:**

* IDE integration
* Code completion
* Interactive coding
* Custom workflows

Aider
-----

AI pair programming in terminal.

**Features:**

* Git integration
* Multiple file editing
* Automatic commits
* Various LLM support
* Voice input

**Usage:**

.. code-block:: bash

    # Start aider
    aider main.py tests.py

    # Give instructions
    > Add input validation to the login function

    # Review changes and commit
    > /commit "Added input validation"

**Commands:**

* ``/add`` - Add files to chat
* ``/commit`` - Commit changes
* ``/undo`` - Undo last change
* ``/test`` - Run test command
* ``/architect`` - Switch to architect mode

**Use Cases:**

* Terminal-based coding
* Git-aware edits
* Rapid prototyping
* Refactoring

GPT-Engineer
------------

Automated codebase generation from prompts.

**Features:**

* Full project generation
* Iterative refinement
* Learning from feedback
* Multiple programming languages

**Usage:**

.. code-block:: bash

    # Create project
    gpt-engineer projects/my-app

    # Provide prompt
    > Create a Flask REST API with user authentication

**Workflow:**

1. Clarification questions
2. Generate plan
3. Implement code
4. Refine based on feedback

**Use Cases:**

* Rapid prototyping
* Project scaffolding
* Learning tool
* Proof of concepts

MetaGPT
-------

Multi-agent software company simulation.

**Features:**

* Role-based agents (PM, Architect, Engineer, QA)
* Document-driven development
* Standard operating procedures
* Complete project generation

**Example:**

.. code-block:: python

    from metagpt.software_company import SoftwareCompany

    company = SoftwareCompany()
    company.invest(investment=3.0)

    result = await company.run(
        idea="Create a CLI tool for managing TODO lists"
    )

**Roles:**

* Product Manager: Requirements
* Architect: System design
* Engineer: Implementation
* QA Engineer: Test plans

**Use Cases:**

* Complete project generation
* Understanding software processes
* Team collaboration simulation
* Education

Specialized Tools
=================

Code Generation
---------------

Codex (OpenAI)
~~~~~~~~~~~~~~

Underlying model for GitHub Copilot.

**Features:**

* Natural language to code
* Multiple languages
* Context-aware
* Fine-tuned for code

StarCoder
~~~~~~~~~

Open-source code generation model.

**Models:**

* StarCoder: 15B parameters
* StarCoder2: Enhanced version
* Multiple size variants

**Features:**

* Trained on The Stack
* Commercial-friendly license
* Multiple languages
* Fillininthe-middle

CodeLlama (Meta)
~~~~~~~~~~~~~~~~

Llama 2 fine-tuned for code.

**Variants:**

* CodeLlama-Base: Foundation
* CodeLlama-Python: Python-specialized
* CodeLlama-Instruct: Instruction-tuned

**Sizes:** 7B, 13B, 34B, 70B parameters

Code Analysis
-------------

tree-sitter
~~~~~~~~~~~

Parser generator for syntax trees.

**Use Cases:**

* AST generation
* Syntax highlighting
* Code navigation
* Refactoring tools

**Example:**

.. code-block:: python

    from tree_sitter import Language, Parser

    parser = Parser()
    parser.set_language(Language('tree-sitter-python.so', 'python'))

    tree = parser.parse(code.encode())
    root_node = tree.root_node

Semgrep
~~~~~~~

Static analysis tool for code patterns.

**Features:**

* Pattern-based search
* Security scanning
* Custom rules
* Multiple languages

**Example:**

.. code-block:: yaml

    rules:
      - id: hardcoded-secret
        pattern: password = "..."
        message: Hardcoded password detected
        severity: ERROR

Testing Frameworks
------------------

Pytest
~~~~~~

Python testing framework.

**Agent Integration:**

* Generate test cases
* Auto-fix failing tests
* Coverage analysis

**Example:**

.. code-block:: python

    # Agent-generated test
    def test_calculate_total():
        assert calculate_total([1, 2, 3]) == 6
        assert calculate_total([]) == 0
        assert calculate_total([-1, 1]) == 0

Evaluation & Benchmarking
--------------------------

EvalPlus
~~~~~~~~

Enhanced code evaluation framework.

**Features:**

* Extended HumanEval
* Comprehensive test cases
* Multiple languages
* Mutation testing

lm-evaluation-harness
~~~~~~~~~~~~~~~~~~~~~

Unified evaluation framework.

**Features:**

* Multiple benchmarks
* Standard metrics
* Reproducible evaluation
* Custom tasks

Infrastructure & Deployment
============================

Docker
------

Containerization for agent environments.

**Use Cases:**

* Isolated code execution
* Reproducible environments
* Sandboxing
* Deployment

Ray
---

Distributed computing framework.

**Features:**

* Ray Serve: Model serving
* Ray Tune: Hyperparameter tuning
* Ray RLlib: RL for agents
* Distributed execution

**Example:**

.. code-block:: python

    import ray

    @ray.remote
    def process_file(file_path):
        # Agent processes file
        return analysis

    # Parallel processing
    results = ray.get([
        process_file.remote(f) for f in files
    ])

Kubernetes
----------

Container orchestration for scaling agents.

**Use Cases:**

* Production deployment
* Auto-scaling
* Load balancing
* High availability

Framework Comparison
====================

General Purpose
---------------

=============  ============  ===============  ===========  =============
Framework      Complexity    Flexibility      Community    Best For
=============  ============  ===============  ===========  =============
LangChain      Medium        Very High        Large        Prototyping
LangGraph      High          Very High        Growing      Production
LlamaIndex     Low           Medium           Large        RAG/Search
AutoGen        Medium        High             Growing      Multi-agent
CrewAI         Low           Medium           Medium       Role-based
=============  ============  ===============  ===========  =============

Code-Specific
-------------

================  ===========  ============  =============
Tool              Integration  Automation    Learning Curve
================  ===========  ============  =============
Continue.dev      IDE          Low           Low
Aider             Terminal     Medium        Low
GPT-Engineer      Standalone   High          Low
MetaGPT           Standalone   Very High     Medium
================  ===========  ============  =============

Selection Guide
===============

Choose Based On
---------------

**For Rapid Prototyping:**

* LangChain
* LlamaIndex
* CrewAI

**For Production Systems:**

* LangGraph
* Semantic Kernel
* Haystack

**For Multi-Agent:**

* AutoGen
* CrewAI
* MetaGPT

**For IDE Integration:**

* Continue.dev
* Custom LangChain integration

**For Terminal Workflow:**

* Aider
* Custom scripts with frameworks

**For Learning:**

* LangChain (extensive docs)
* AutoGen (good examples)
* CrewAI (simple concepts)

Integration Patterns
====================

Combining Frameworks
--------------------

**LangChain + LlamaIndex:**

.. code-block:: python

    from langchain.agents import Tool
    from llama_index import VectorStoreIndex

    # LlamaIndex for retrieval
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()

    # LangChain for orchestration
    tools = [
        Tool(
            name="CodeSearch",
            func=lambda q: query_engine.query(q),
            description="Search codebase"
        )
    ]

    agent = initialize_agent(tools, llm)

**Framework + Custom Tools:**

Most frameworks support custom tool integration:

.. code-block:: python

    # Custom tool
    def my_custom_tool(input: str) -> str:
        # Your logic
        return result

    # Wrap for framework
    tool = Tool(
        name="CustomTool",
        func=my_custom_tool,
        description="Description"
    )

Best Practices
==============

1. **Start Simple:** Begin with basic framework features
2. **Understand Abstractions:** Know what framework does for you
3. **Read Docs:** Framework docs are essential
4. **Check Examples:** Learn from official examples
5. **Community Support:** Use active frameworks
6. **Version Pinning:** Lock framework versions
7. **Testing:** Test framework integrations thoroughly
8. **Monitoring:** Add observability
9. **Error Handling:** Robust error management
10. **Stay Updated:** Frameworks evolve rapidly

Resources
=========

Documentation
-------------

* LangChain: https://python.langchain.com
* LlamaIndex: https://docs.llamaindex.ai
* AutoGen: https://microsoft.github.io/autogen
* CrewAI: https://docs.crewai.com

Communities
-----------

* Discord servers for major frameworks
* GitHub discussions
* Reddit communities
* Twitter/X communities

Learning Resources
------------------

* Official tutorials
* YouTube channels
* Course platforms (Coursera, Udemy)
* Blog posts and articles

See Also
========

* :doc:`patterns`
* :doc:`mcp`
* :doc:`a2a`
* :doc:`../llm/tool_selection`
* :doc:`../deployments`
