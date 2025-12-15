====================
Latency
====================

Overview
========

Latency is the time delay between a user's request and the agent's response. For coding agents, low latency is crucial for providing a good user experience, especially in interactive scenarios.

Why Latency Matters
===================

User Experience
---------------

* **Interactive Sessions:** Real-time coding assistance requires low latency
* **Flow State:** High latency interrupts developer flow
* **Productivity:** Faster responses mean more iterations per hour
* **Adoption:** Users abandon slow tools

**Acceptable Latency Ranges:**

* **Excellent:** <1 second
* **Good:** 1-3 seconds
* **Acceptable:** 3-10 seconds
* **Poor:** >10 seconds

Types of Latency
================

End-to-End Latency
------------------

Total time from request to complete response.

.. code-block:: text

    User Request → Processing → LLM Inference → Post-processing → Response

**Measurement:**

.. code-block:: python

    import time

    async def measure_e2e_latency(agent: Agent, task: str) -> float:
        start = time.time()
        result = await agent.execute(task)
        end = time.time()
        return end - start

Time to First Token (TTFT)
---------------------------

Time until first response token appears.

**Importance:**

* User perceives system as responsive
* Enables streaming responses
* Critical for interactive experiences

**Measurement:**

.. code-block:: python

    async def measure_ttft(llm: LLM, prompt: str) -> float:
        start = time.time()

        async for chunk in llm.stream(prompt):
            ttft = time.time() - start
            return ttft  # Return after first token

Inter-Token Latency (ITL)
--------------------------

Time between successive tokens.

**Impact:**

* Affects streaming smoothness
* User perception of speed
* Overall completion time

**Measurement:**

.. code-block:: python

    async def measure_itl(llm: LLM, prompt: str) -> List[float]:
        intervals = []
        last_time = time.time()

        async for chunk in llm.stream(prompt):
            current_time = time.time()
            intervals.append(current_time - last_time)
            last_time = current_time

        return intervals

Latency Components
==================

Network Latency
---------------

Time spent in network communication.

**Factors:**

* Geographic distance
* Network quality
* Protocol overhead
* Payload size

**Optimization:**

.. code-block:: python

    # Use compression
    import gzip

    def compress_request(data: dict) -> bytes:
        json_str = json.dumps(data)
        return gzip.compress(json_str.encode())

    # Regional endpoints
    def get_nearest_endpoint(user_location: str) -> str:
        endpoints = {
            "us-west": "api-us-west.example.com",
            "us-east": "api-us-east.example.com",
            "eu": "api-eu.example.com",
            "asia": "api-asia.example.com"
        }
        return endpoints.get(user_location, endpoints["us-west"])

LLM Inference Latency
---------------------

Time for model to generate response.

**Factors:**

* Model size (7B, 13B, 70B parameters)
* Context length
* Generation length
* Hardware (GPU type, memory)
* Batch size

**Typical Latencies:**

.. code-block:: text

    Model Size    |  Hardware      |  Latency (per token)
    --------------------------------------------------------
    7B params     |  A100 (40GB)   |  ~20ms
    13B params    |  A100 (40GB)   |  ~30ms
    34B params    |  A100 (80GB)   |  ~50ms
    70B params    |  A100 (80GB)   |  ~80ms
    GPT-4         |  API           |  ~50-100ms

Tool Execution Latency
----------------------

Time for external tools to execute.

**Examples:**

* File I/O: 10-100ms
* Code execution: 100ms-5s
* Web search: 1-3s
* Database queries: 10-500ms

**Optimization:**

.. code-block:: python

    import asyncio

    async def parallel_tool_execution(tools: List[Tool], inputs: List[Any]):
        """Execute multiple tools in parallel."""
        tasks = [tool.execute(input) for tool, input in zip(tools, inputs)]
        results = await asyncio.gather(*tasks)
        return results

Processing Latency
------------------

Time for pre/post-processing.

**Components:**

* Input parsing
* Context retrieval
* Output formatting
* Validation

Measuring Latency
=================

Metrics
-------

Percentiles
~~~~~~~~~~~

More informative than averages.

.. code-block:: python

    import numpy as np

    def compute_latency_percentiles(latencies: List[float]):
        return {
            "p50": np.percentile(latencies, 50),  # Median
            "p90": np.percentile(latencies, 90),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "max": max(latencies),
            "mean": np.mean(latencies)
        }

Throughput
~~~~~~~~~~

Requests processed per second.

.. code-block:: python

    def compute_throughput(num_requests: int, total_time: float) -> float:
        return num_requests / total_time

Monitoring
----------

Real-Time Monitoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from prometheus_client import Histogram

    request_latency = Histogram(
        'agent_request_duration_seconds',
        'Agent request latency',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    )

    @request_latency.time()
    async def execute_agent(task: str):
        return await agent.execute(task)

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    async def execute_with_tracing(task: str):
        with tracer.start_as_current_span("agent_execution") as span:
            with tracer.start_as_current_span("llm_inference"):
                result = await llm.generate(task)

            with tracer.start_as_current_span("post_processing"):
                formatted = format_result(result)

            return formatted

Reducing Latency
================

Caching
-------

Response Caching
~~~~~~~~~~~~~~~~

.. code-block:: python

    import redis
    from functools import wraps

    redis_client = redis.Redis()

    def cache_response(ttl: int = 3600):
        def decorator(func):
            @wraps(func)
            async def wrapper(task: str, *args, **kwargs):
                cache_key = f"agent:response:{hash(task)}"

                # Check cache
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)

                # Generate response
                response = await func(task, *args, **kwargs)

                # Store in cache
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response)
                )

                return response
            return wrapper
        return decorator

Embedding Caching
~~~~~~~~~~~~~~~~~

Cache computed embeddings.

.. code-block:: python

    class EmbeddingCache:
        def __init__(self):
            self.cache = {}

        async def get_embedding(self, text: str, model):
            key = (text, model.name)

            if key not in self.cache:
                self.cache[key] = await model.embed(text)

            return self.cache[key]

Streaming
---------

Stream responses as they're generated.

.. code-block:: python

    async def stream_response(agent: Agent, task: str):
        """Stream agent response token by token."""
        async for chunk in agent.stream_execute(task):
            yield chunk

Model Optimization
------------------

Quantization
~~~~~~~~~~~~

Reduce model precision for faster inference.

.. code-block:: python

    from transformers import AutoModelForCausalLM

    # Load 4-bit quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "model-name",
        load_in_4bit=True,
        device_map="auto"
    )

Speculative Decoding
~~~~~~~~~~~~~~~~~~~~

Use smaller model to speedup larger model.

.. code-block:: python

    async def speculative_decoding(
        draft_model: LLM,
        target_model: LLM,
        prompt: str
    ):
        """Use draft model to speculate, verify with target."""
        # Draft model generates candidates quickly
        candidates = await draft_model.generate(
            prompt,
            n=5,
            max_tokens=10
        )

        # Target model verifies
        verified = await target_model.verify(candidates)

        return verified

Batching
~~~~~~~~

Process multiple requests together.

.. code-block:: python

    class BatchProcessor:
        def __init__(self, batch_size: int = 8, timeout: float = 0.1):
            self.batch_size = batch_size
            self.timeout = timeout
            self.queue = asyncio.Queue()

        async def add_request(self, request):
            future = asyncio.Future()
            await self.queue.put((request, future))
            return await future

        async def process_batches(self):
            while True:
                batch = []
                deadline = asyncio.get_event_loop().time() + self.timeout

                # Collect batch
                while len(batch) < self.batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    # Process batch
                    requests = [item[0] for item in batch]
                    futures = [item[1] for item in batch]

                    results = await self.process_batch(requests)

                    # Return results
                    for future, result in zip(futures, results):
                        future.set_result(result)

Prompt Optimization
-------------------

Reduce prompt length while maintaining quality.

.. code-block:: python

    def optimize_prompt(prompt: str, max_tokens: int = 2000) -> str:
        """Compress prompt to reduce latency."""
        if count_tokens(prompt) <= max_tokens:
            return prompt

        # Remove redundant information
        prompt = remove_redundancy(prompt)

        # Summarize if still too long
        if count_tokens(prompt) > max_tokens:
            prompt = summarize(prompt, max_tokens)

        return prompt

Parallel Processing
-------------------

Execute independent operations concurrently.

.. code-block:: python

    async def parallel_agent_workflow(task: str):
        """Execute agent workflow with parallel operations."""
        # These can run in parallel
        code_search, doc_search, recent_context = await asyncio.gather(
            search_codebase(task),
            search_documentation(task),
            get_recent_context()
        )

        # Combine and proceed
        context = combine_context(code_search, doc_search, recent_context)

        # Generate solution
        solution = await generate_solution(task, context)

        return solution

Infrastructure
--------------

Load Balancing
~~~~~~~~~~~~~~

Distribute requests across instances.

Regional Deployment
~~~~~~~~~~~~~~~~~~~

Deploy close to users.

CDN
~~~

Cache static assets.

Warm Instances
~~~~~~~~~~~~~~

Keep instances ready.

.. code-block:: python

    class WarmInstancePool:
        def __init__(self, pool_size: int = 5):
            self.pool = asyncio.Queue(maxsize=pool_size)
            self.initialize_pool()

        async def initialize_pool(self):
            """Pre-initialize agent instances."""
            for _ in range(self.pool.maxsize):
                agent = await initialize_agent()
                await self.pool.put(agent)

        async def get_agent(self):
            """Get warm agent from pool."""
            agent = await self.pool.get()

            # Asynchronously create replacement
            asyncio.create_task(self.add_to_pool())

            return agent

        async def add_to_pool(self):
            agent = await initialize_agent()
            await self.pool.put(agent)

Latency Budgets
===============

Setting Budgets
---------------

Allocate latency across components.

.. code-block:: python

    @dataclass
    class LatencyBudget:
        total: float = 3.0  # 3 seconds total
        network: float = 0.1  # 100ms
        preprocessing: float = 0.2  # 200ms
        llm_inference: float = 2.0  # 2 seconds
        tool_execution: float = 0.5  # 500ms
        postprocessing: float = 0.2  # 200ms

        def validate(self):
            """Ensure components sum to total."""
            components = (
                self.network +
                self.preprocessing +
                self.llm_inference +
                self.tool_execution +
                self.postprocessing
            )
            assert components <= self.total

Monitoring Budgets
------------------

.. code-block:: python

    class LatencyBudgetMonitor:
        def __init__(self, budget: LatencyBudget):
            self.budget = budget
            self.violations = []

        def check_component(
            self,
            component: str,
            actual_latency: float
        ):
            budget = getattr(self.budget, component)

            if actual_latency > budget:
                self.violations.append({
                    "component": component,
                    "budget": budget,
                    "actual": actual_latency,
                    "overage": actual_latency - budget,
                    "timestamp": datetime.now()
                })

Trade-offs
==========

Latency vs. Accuracy
--------------------

* Faster models may be less accurate
* Can use fast first-pass, refine if needed
* Complexity-based routing

Latency vs. Cost
----------------

* Lower latency often costs more
* Premium for faster GPUs
* Caching reduces both

Latency vs. Throughput
----------------------

* Batching increases throughput but latency
* Balance based on use case

Best Practices
==============

1. **Measure Everything:** Comprehensive latency tracking
2. **Set Budgets:** Define acceptable latencies
3. **Optimize Hotspots:** Focus on slowest components
4. **Stream When Possible:** Reduce perceived latency
5. **Cache Aggressively:** Repeated queries
6. **Monitor Continuously:** Track p95, p99
7. **Regional Deployment:** Close to users
8. **Async Processing:** Non-blocking operations
9. **Fail Fast:** Timeouts on slow operations
10. **User Feedback:** Loading indicators, progress

Resources
=========

Tools
-----

* Prometheus: Metrics collection
* Grafana: Visualization
* Jaeger/Zipkin: Distributed tracing
* Load testing: Locust, k6

See Also
========

* :doc:`accuracy`
* :doc:`cost`
* :doc:`../deployments`
* :doc:`../evaluations`
