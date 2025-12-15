====================
Memory Management
====================

Overview
========

Effective memory management is crucial for coding agents to maintain context, learn from experience, and provide coherent assistance across sessions. This section covers memory types, storage strategies, and implementation patterns.

Why Memory Matters
==================

Challenges Without Memory
-------------------------

* Loss of conversation context
* Repeating the same mistakes
* No learning from experience
* Inability to maintain long-term relationships
* Poor personalization

Benefits of Memory
------------------

* Context preservation across turns
* Learning from past interactions
* Personalized responses
* Improved efficiency
* Better user experience

Memory Types
============

Short-Term Memory (STM)
-----------------------

Immediate context within current session.

**Characteristics:**

* Limited capacity
* High access speed
* Volatile (session-bound)
* Full detail retention

**Contents:**

* Current conversation
* Active code context
* Recent tool outputs
* Working variables

**Implementation:**

.. code-block:: python

    class ShortTermMemory:
        def __init__(self, max_tokens=8000):
            self.messages = []
            self.max_tokens = max_tokens

        def add_message(self, role: str, content: str):
            self.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            })

            # Trim if exceeds limit
            while self.token_count() > self.max_tokens:
                self.messages.pop(0)

        def get_context(self) -> List[Dict]:
            return self.messages

        def token_count(self) -> int:
            # Calculate total tokens
            return sum(len(m["content"]) // 4 for m in self.messages)

Long-Term Memory (LTM)
----------------------

Persistent knowledge across sessions.

**Characteristics:**

* Unlimited capacity (practically)
* Slower access (database queries)
* Persistent storage
* Indexed for retrieval

**Contents:**

* Past conversations
* User preferences
* Project knowledge
* Learned patterns
* Historical decisions

**Storage Options:**

* Vector databases (Pinecone, Weaviate, Chroma)
* Graph databases (Neo4j)
* Relational databases (PostgreSQL)
* Document stores (MongoDB)

**Implementation:**

.. code-block:: python

    class LongTermMemory:
        def __init__(self, vector_db):
            self.db = vector_db
            self.embedding_model = EmbeddingModel()

        async def store(self, content: str, metadata: Dict):
            embedding = await self.embedding_model.embed(content)

            await self.db.upsert({
                "id": generate_id(),
                "embedding": embedding,
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now()
            })

        async def retrieve(
            self,
            query: str,
            limit: int = 5,
            filters: Dict = None
        ) -> List[Dict]:
            query_embedding = await self.embedding_model.embed(query)

            results = await self.db.search(
                embedding=query_embedding,
                limit=limit,
                filters=filters
            )

            return results

Working Memory
--------------

Active task state and context.

**Characteristics:**

* Task-scoped
* Mutable state
* Structured data
* Cleared on task completion

**Contents:**

* Current goal/task
* Intermediate results
* Active variables
* Execution state
* Sub-goals

**Implementation:**

.. code-block:: python

    @dataclass
    class WorkingMemory:
        current_goal: str
        sub_goals: List[str]
        variables: Dict[str, Any]
        results: List[Any]
        state: Dict[str, Any]

        def push_goal(self, goal: str):
            self.sub_goals.append(goal)

        def pop_goal(self) -> str:
            return self.sub_goals.pop() if self.sub_goals else None

        def set_variable(self, key: str, value: Any):
            self.variables[key] = value

        def get_variable(self, key: str) -> Any:
            return self.variables.get(key)

        def clear(self):
            self.sub_goals.clear()
            self.variables.clear()
            self.results.clear()
            self.state.clear()

Episodic Memory
---------------

Memory of specific past experiences.

**Characteristics:**

* Event-based
* Temporal ordering
* Contextual details
* Searchable by similarity

**Contents:**

* Past debugging sessions
* Feature implementations
* Refactoring episodes
* Problem-solution pairs

**Structure:**

.. code-block:: python

    @dataclass
    class Episode:
        id: str
        task: str
        context: Dict
        actions: List[Action]
        outcome: str
        success: bool
        duration: float
        timestamp: datetime
        learned: List[str]
        tags: List[str]

    class EpisodicMemory:
        def __init__(self, storage):
            self.storage = storage

        async def store_episode(self, episode: Episode):
            await self.storage.insert(episode)

        async def recall_similar(
            self,
            task: str,
            limit: int = 3
        ) -> List[Episode]:
            # Find similar past episodes
            return await self.storage.search_similar(task, limit)

        async def get_successful_patterns(
            self,
            task_type: str
        ) -> List[Episode]:
            # Retrieve successful patterns for task type
            return await self.storage.filter(
                task_type=task_type,
                success=True
            )

Semantic Memory
---------------

General knowledge and facts.

**Characteristics:**

* Decontextualized knowledge
* Generalizations
* Facts and rules
* Schema and patterns

**Contents:**

* Code patterns
* Best practices
* API knowledge
* Language syntax
* Design patterns

**Implementation:**

.. code-block:: python

    class SemanticMemory:
        def __init__(self):
            self.facts = {}
            self.patterns = []
            self.rules = []

        def add_fact(self, key: str, value: Any):
            self.facts[key] = value

        def get_fact(self, key: str) -> Any:
            return self.facts.get(key)

        def add_pattern(self, pattern: CodePattern):
            self.patterns.append(pattern)

        def find_patterns(self, criteria: Dict) -> List[CodePattern]:
            return [p for p in self.patterns if p.matches(criteria)]

Procedural Memory
-----------------

Knowledge of how to do things.

**Characteristics:**

* Skill-based
* Automatic execution
* Learned procedures
* Efficient retrieval

**Contents:**

* Common workflows
* Tool usage patterns
* Problem-solving strategies
* Refactoring procedures

Memory Storage
==============

Vector Databases
----------------

Store and retrieve embeddings for semantic search.

Pinecone
~~~~~~~~

Cloud-native vector database.

**Features:**

* Managed service
* High performance
* Hybrid search
* Metadata filtering

**Example:**

.. code-block:: python

    import pinecone

    pinecone.init(api_key="...", environment="...")
    index = pinecone.Index("code-memories")

    # Store
    index.upsert(vectors=[{
        "id": "mem-1",
        "values": embedding,
        "metadata": {
            "content": "...",
            "type": "conversation",
            "timestamp": "..."
        }
    }])

    # Query
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

Weaviate
~~~~~~~~

Open-source vector database with AI-native features.

**Features:**

* Self-hosted or cloud
* GraphQL API
* Hybrid search
* Module ecosystem

**Example:**

.. code-block:: python

    import weaviate

    client = weaviate.Client("http://localhost:8080")

    # Create schema
    schema = {
        "class": "CodeMemory",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "context", "dataType": ["text"]},
            {"name": "timestamp", "dataType": ["date"]}
        ]
    }
    client.schema.create_class(schema)

    # Store
    client.data_object.create({
        "content": "...",
        "context": "...",
        "timestamp": "..."
    }, "CodeMemory")

    # Query
    result = client.query.get("CodeMemory", ["content", "context"])\
        .with_near_text({"concepts": ["authentication"]})\
        .with_limit(5)\
        .do()

Chroma
~~~~~~

Embeddable vector database for AI applications.

**Features:**

* Lightweight
* Easy integration
* Local or client-server
* Open-source

**Example:**

.. code-block:: python

    import chromadb

    client = chromadb.Client()
    collection = client.create_collection("code_memories")

    # Add
    collection.add(
        embeddings=[embedding],
        documents=["content"],
        metadatas=[{"type": "debug", "file": "main.py"}],
        ids=["mem-1"]
    )

    # Query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

Qdrant
~~~~~~

High-performance vector search engine.

**Features:**

* Fast search
* Filtering capabilities
* Distributed mode
* Rust-based performance

Graph Databases
---------------

Store relationships and connections.

Neo4j
~~~~~

Leading graph database.

**Use Cases:**

* Code dependency graphs
* Knowledge graphs
* Relationship tracking
* Connected knowledge

**Example:**

.. code-block:: python

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

    with driver.session() as session:
        # Store relationship
        session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (t:Task {id: $task_id})
            MERGE (u)-[:WORKED_ON {date: $date}]->(t)
        """, user_id="...", task_id="...", date="...")

        # Query
        result = session.run("""
            MATCH (u:User)-[:WORKED_ON]->(t:Task)
            WHERE t.type = $type
            RETURN t
        """, type="debug")

Relational Databases
--------------------

Traditional structured storage.

PostgreSQL with pgvector
~~~~~~~~~~~~~~~~~~~~~~~~

Add vector search to PostgreSQL.

**Example:**

.. code-block:: sql

    -- Create extension
    CREATE EXTENSION vector;

    -- Create table
    CREATE TABLE code_memories (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(1536),
        metadata JSONB,
        created_at TIMESTAMP
    );

    -- Create index
    CREATE INDEX ON code_memories
    USING ivfflat (embedding vector_cosine_ops);

    -- Insert
    INSERT INTO code_memories (content, embedding, metadata)
    VALUES ('...', '[0.1, 0.2, ...]', '{"type": "code"}');

    -- Search
    SELECT content, metadata
    FROM code_memories
    ORDER BY embedding <=> '[0.1, 0.2, ...]'
    LIMIT 5;

Memory Strategies
=================

Context Window Management
-------------------------

Sliding Window
~~~~~~~~~~~~~~

Keep recent N messages/tokens.

.. code-block:: python

    class SlidingWindowMemory:
        def __init__(self, window_size: int = 10):
            self.messages = deque(maxlen=window_size)

        def add(self, message):
            self.messages.append(message)

        def get_all(self):
            return list(self.messages)

Token-Based Truncation
~~~~~~~~~~~~~~~~~~~~~~

Maintain token budget.

.. code-block:: python

    class TokenBudgetMemory:
        def __init__(self, max_tokens: int = 4000):
            self.messages = []
            self.max_tokens = max_tokens

        def add(self, message):
            self.messages.append(message)
            self._trim_to_budget()

        def _trim_to_budget(self):
            while self.token_count() > self.max_tokens:
                self.messages.pop(0)

Summarization
~~~~~~~~~~~~~

Compress older context.

.. code-block:: python

    class SummarizingMemory:
        def __init__(self, llm, summary_threshold: int = 20):
            self.llm = llm
            self.messages = []
            self.summary = ""
            self.summary_threshold = summary_threshold

        async def add(self, message):
            self.messages.append(message)

            if len(self.messages) >= self.summary_threshold:
                await self._summarize()

        async def _summarize(self):
            # Summarize older messages
            old_messages = self.messages[:10]
            summary = await self.llm.summarize(old_messages)

            # Update
            self.summary += f"\n{summary}"
            self.messages = self.messages[10:]

        def get_context(self):
            return f"Summary: {self.summary}\n\nRecent:\n" + \
                   "\n".join(self.messages)

Hierarchical Compression
~~~~~~~~~~~~~~~~~~~~~~~~

Multiple detail levels.

.. code-block:: python

    class HierarchicalMemory:
        def __init__(self, llm):
            self.llm = llm
            self.level1 = []  # Full detail
            self.level2 = []  # Medium summary
            self.level3 = []  # High-level summary

        async def add(self, message):
            self.level1.append(message)

            if len(self.level1) > 10:
                summary = await self.llm.summarize(self.level1[:5])
                self.level2.append(summary)
                self.level1 = self.level1[5:]

            if len(self.level2) > 10:
                high_level = await self.llm.summarize(self.level2[:5])
                self.level3.append(high_level)
                self.level2 = self.level2[5:]

Retrieval Strategies
--------------------

Semantic Search
~~~~~~~~~~~~~~~

Find relevant memories by meaning.

.. code-block:: python

    async def semantic_retrieval(
        query: str,
        memory: LongTermMemory,
        top_k: int = 5
    ):
        results = await memory.retrieve(query, limit=top_k)
        return results

Keyword Search
~~~~~~~~~~~~~~

Traditional text matching.

.. code-block:: python

    def keyword_retrieval(
        keywords: List[str],
        memories: List[Memory]
    ):
        matches = []
        for memory in memories:
            if any(kw in memory.content.lower() for kw in keywords):
                matches.append(memory)
        return matches

Hybrid Search
~~~~~~~~~~~~~

Combine semantic and keyword.

.. code-block:: python

    async def hybrid_retrieval(
        query: str,
        keywords: List[str],
        memory: LongTermMemory,
        alpha: float = 0.5
    ):
        semantic_results = await semantic_retrieval(query, memory)
        keyword_results = keyword_retrieval(keywords, memory.get_all())

        # Combine and re-rank
        combined = merge_and_rerank(
            semantic_results,
            keyword_results,
            alpha
        )
        return combined

Temporal Retrieval
~~~~~~~~~~~~~~~~~~

Retrieve by time relevance.

.. code-block:: python

    def temporal_retrieval(
        memories: List[Memory],
        time_window: timedelta
    ):
        now = datetime.now()
        return [
            m for m in memories
            if now - m.timestamp <= time_window
        ]

Relevance Filtering
~~~~~~~~~~~~~~~~~~~

Score and filter by relevance threshold.

.. code-block:: python

    def relevance_filtering(
        results: List[Tuple[Memory, float]],
        threshold: float = 0.7
    ):
        return [
            memory for memory, score in results
            if score >= threshold
        ]

Memory Operations
=================

Consolidation
-------------

Merge similar memories.

.. code-block:: python

    async def consolidate_memories(
        memories: List[Memory],
        similarity_threshold: float = 0.9
    ):
        consolidated = []
        processed = set()

        for i, mem1 in enumerate(memories):
            if i in processed:
                continue

            similar = [mem1]
            for j, mem2 in enumerate(memories[i+1:], start=i+1):
                if similarity(mem1, mem2) > similarity_threshold:
                    similar.append(mem2)
                    processed.add(j)

            # Merge similar memories
            merged = merge_memories(similar)
            consolidated.append(merged)

        return consolidated

Forgetting
----------

Remove obsolete or irrelevant memories.

**Strategies:**

* Time-based decay
* Relevance-based pruning
* Capacity-based eviction
* Manual curation

.. code-block:: python

    class ForgettingMemory:
        def __init__(self, decay_rate: float = 0.95):
            self.memories = {}
            self.decay_rate = decay_rate

        def access(self, memory_id: str):
            # Reinforce accessed memory
            if memory_id in self.memories:
                self.memories[memory_id]["strength"] = 1.0

        def decay(self):
            # Decay all memories
            to_remove = []
            for mem_id, memory in self.memories.items():
                memory["strength"] *= self.decay_rate
                if memory["strength"] < 0.1:
                    to_remove.append(mem_id)

            # Remove weak memories
            for mem_id in to_remove:
                del self.memories[mem_id]

Updating
--------

Modify existing memories with new information.

.. code-block:: python

    async def update_memory(
        memory_id: str,
        new_info: str,
        memory_store: LongTermMemory
    ):
        # Retrieve existing
        existing = await memory_store.get(memory_id)

        # Merge with new information
        updated_content = merge_information(
            existing.content,
            new_info
        )

        # Update embedding and store
        await memory_store.update(memory_id, updated_content)

Code-Specific Memory
====================

Codebase Knowledge
------------------

Store understanding of code structure.

.. code-block:: python

    @dataclass
    class CodebaseMemory:
        file_structure: Dict[str, List[str]]
        functions: Dict[str, FunctionInfo]
        classes: Dict[str, ClassInfo]
        imports: Dict[str, List[str]]
        tests: Dict[str, List[str]]

        def add_file(self, path: str, analysis: FileAnalysis):
            self.file_structure[path] = analysis.structure
            self.functions.update(analysis.functions)
            self.classes.update(analysis.classes)

        def find_function(self, name: str) -> Optional[FunctionInfo]:
            return self.functions.get(name)

User Preferences
----------------

Remember user coding style and preferences.

.. code-block:: python

    @dataclass
    class UserPreferences:
        language_preferences: Dict[str, Any]
        style_guide: Dict[str, Any]
        common_patterns: List[Pattern]
        avoided_patterns: List[Pattern]
        tooling: Dict[str, str]

        def update_from_feedback(self, feedback: Feedback):
            if feedback.type == "style":
                self.style_guide.update(feedback.preferences)
            elif feedback.type == "pattern":
                if feedback.positive:
                    self.common_patterns.append(feedback.pattern)
                else:
                    self.avoided_patterns.append(feedback.pattern)

Problem-Solution Pairs
----------------------

Remember how problems were solved.

.. code-block:: python

    @dataclass
    class ProblemSolution:
        problem: str
        context: Dict
        solution: str
        approach: str
        success: bool
        time_to_solve: float
        tools_used: List[str]

    class ProblemSolutionMemory:
        def __init__(self, storage):
            self.storage = storage

        async def store(self, ps: ProblemSolution):
            await self.storage.insert(ps)

        async def find_similar_solution(
            self,
            problem: str
        ) -> Optional[ProblemSolution]:
            # Find similar past problem
            results = await self.storage.search_similar(problem)
            return results[0] if results else None

Best Practices
==============

Memory Design
-------------

1. **Clear Separation:** Distinguish memory types clearly
2. **Appropriate Storage:** Match storage to access patterns
3. **Efficient Retrieval:** Index for common queries
4. **Decay Strategy:** Plan for forgetting
5. **Privacy Aware:** Handle sensitive data properly

Implementation
--------------

1. **Start Simple:** Begin with basic short-term memory
2. **Add Gradually:** Introduce complexity as needed
3. **Monitor Usage:** Track memory access patterns
4. **Optimize Queries:** Profile and optimize slow retrievals
5. **Test Thoroughly:** Verify retrieval accuracy

Maintenance
-----------

1. **Regular Cleanup:** Remove stale data
2. **Monitor Size:** Track storage growth
3. **Backup Data:** Protect against loss
4. **Version Schemas:** Plan for schema evolution
5. **Audit Quality:** Review memory contents

Challenges
==========

Common Issues
-------------

* **Context Overflow:** Exceeding token limits
* **Retrieval Irrelevance:** Poor search results
* **Stale Data:** Outdated information
* **Inconsistency:** Conflicting memories
* **Privacy Concerns:** Sensitive data retention

Solutions
---------

* Effective summarization
* Hybrid search strategies
* Automatic expiration
* Conflict resolution mechanisms
* Data sanitization

Resources
=========

Vector Databases
----------------

* Pinecone: https://www.pinecone.io
* Weaviate: https://weaviate.io
* Chroma: https://www.trychroma.com
* Qdrant: https://qdrant.tech

Memory Frameworks
-----------------

* LangChain Memory
* LlamaIndex Memory
* Mem0
* Zep

Papers
------

[Add relevant papers on memory for AI agents]

See Also
========

* :doc:`llm/multi_turn`
* :doc:`tools/frameworks`
* :doc:`evaluations`
