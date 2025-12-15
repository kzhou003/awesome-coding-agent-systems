====================
Cost
====================

Overview
========

Cost is a critical factor in deploying coding agents at scale. Understanding and optimizing costs ensures sustainable operations and competitive pricing.

Cost Components
===============

LLM API Costs
-------------

Primary cost driver for most agents.

**Pricing Models:**

.. code-block:: text

    Provider      |  Model          |  Input (per 1M tokens)  |  Output
    --------------------------------------------------------------------
    OpenAI        |  GPT-4 Turbo    |  $10                    |  $30
    OpenAI        |  GPT-3.5 Turbo  |  $0.50                  |  $1.50
    Anthropic     |  Claude 3 Opus  |  $15                    |  $75
    Anthropic     |  Claude 3 Sonnet|  $3                     |  $15
    Google        |  Gemini Pro     |  $0.50                  |  $1.50

**Calculation:**

.. code-block:: python

    @dataclass
    class LLMCost:
        input_tokens: int
        output_tokens: int
        input_price_per_million: float
        output_price_per_million: float

        def total_cost(self) -> float:
            input_cost = (self.input_tokens / 1_000_000) * self.input_price_per_million
            output_cost = (self.output_tokens / 1_000_000) * self.output_price_per_million
            return input_cost + output_cost

    # Example
    cost = LLMCost(
        input_tokens=2000,
        output_tokens=500,
        input_price_per_million=10.0,  # GPT-4
        output_price_per_million=30.0
    )
    print(f"Cost: ${cost.total_cost():.4f}")  # $0.0350

Infrastructure Costs
--------------------

**Self-Hosted Models:**

* GPU rental (A100: $1-3/hour)
* CPU instances
* Storage
* Network bandwidth
* Maintenance

**Cloud Services:**

* API Gateway
* Load balancers
* Databases
* Caching (Redis)
* Monitoring

Tool & API Costs
----------------

* Code execution sandboxes
* Search APIs
* External services
* Database queries
* Vector database operations

Development & Operations
------------------------

* Developer time
* DevOps resources
* Monitoring tools
* Testing infrastructure

Cost Tracking
=============

Request-Level Tracking
----------------------

.. code-block:: python

    @dataclass
    class RequestCost:
        request_id: str
        timestamp: datetime
        llm_cost: float
        tool_costs: Dict[str, float]
        infrastructure_cost: float
        user_id: str

        def total_cost(self) -> float:
            return (
                self.llm_cost +
                sum(self.tool_costs.values()) +
                self.infrastructure_cost
            )

    class CostTracker:
        def __init__(self):
            self.costs = []

        def track_request(self, cost: RequestCost):
            self.costs.append(cost)

        def total_cost(
            self,
            start: datetime = None,
            end: datetime = None
        ) -> float:
            filtered = self.costs
            if start:
                filtered = [c for c in filtered if c.timestamp >= start]
            if end:
                filtered = [c for c in filtered if c.timestamp <= end]

            return sum(c.total_cost() for c in filtered)

        def cost_by_user(self) -> Dict[str, float]:
            user_costs = defaultdict(float)
            for cost in self.costs:
                user_costs[cost.user_id] += cost.total_cost()
            return dict(user_costs)

User-Level Analytics
--------------------

.. code-block:: python

    class UserCostAnalytics:
        def __init__(self, tracker: CostTracker):
            self.tracker = tracker

        def average_cost_per_request(self, user_id: str) -> float:
            user_costs = [
                c for c in self.tracker.costs
                if c.user_id == user_id
            ]
            if not user_costs:
                return 0.0

            total = sum(c.total_cost() for c in user_costs)
            return total / len(user_costs)

        def high_cost_users(self, threshold: float = 10.0) -> List[str]:
            cost_by_user = self.tracker.cost_by_user()
            return [
                user for user, cost in cost_by_user.items()
                if cost >= threshold
            ]

Cost Optimization
=================

Model Selection
---------------

Choose appropriate model for task complexity.

.. code-block:: python

    class AdaptiveModelRouter:
        def __init__(self):
            self.models = {
                "simple": {
                    "name": "gpt-3.5-turbo",
                    "cost_per_million_input": 0.5,
                    "cost_per_million_output": 1.5
                },
                "complex": {
                    "name": "gpt-4-turbo",
                    "cost_per_million_input": 10.0,
                    "cost_per_million_output": 30.0
                }
            }

        def select_model(self, task: str) -> str:
            complexity = self.estimate_complexity(task)

            if complexity < 0.5:
                return "simple"
            else:
                return "complex"

        def estimate_complexity(self, task: str) -> float:
            # Simple heuristic
            indicators = {
                "algorithm": 0.8,
                "optimize": 0.7,
                "refactor large": 0.9,
                "fix typo": 0.1,
                "add comment": 0.1
            }

            task_lower = task.lower()
            scores = [
                score for keyword, score in indicators.items()
                if keyword in task_lower
            ]

            return max(scores) if scores else 0.5

Prompt Optimization
-------------------

Reduce token usage without sacrificing quality.

.. code-block:: python

    def optimize_prompt(prompt: str) -> str:
        """Reduce prompt tokens while maintaining meaning."""
        # Remove unnecessary whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()

        # Remove redundant instructions
        prompt = remove_repetitions(prompt)

        # Use abbreviations where clear
        prompt = apply_abbreviations(prompt)

        return prompt

    def remove_repetitions(text: str) -> str:
        """Remove repeated instructions."""
        lines = text.split('\n')
        seen = set()
        unique_lines = []

        for line in lines:
            normalized = line.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)

        return '\n'.join(unique_lines)

Caching
-------

Avoid redundant API calls.

.. code-block:: python

    class CostAwareCache:
        def __init__(self, redis_client):
            self.redis = redis_client
            self.cache_hits = 0
            self.cache_misses = 0
            self.cost_saved = 0.0

        async def get_or_compute(
            self,
            key: str,
            compute_func,
            estimated_cost: float,
            ttl: int = 3600
        ):
            # Check cache
            cached = self.redis.get(key)

            if cached:
                self.cache_hits += 1
                self.cost_saved += estimated_cost
                return json.loads(cached)

            # Cache miss - compute
            self.cache_misses += 1
            result = await compute_func()

            # Store
            self.redis.setex(key, ttl, json.dumps(result))

            return result

        def cache_stats(self) -> Dict:
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total if total > 0 else 0

            return {
                "hit_rate": hit_rate,
                "cost_saved": self.cost_saved,
                "total_requests": total
            }

Response Streaming
------------------

Stop generation when enough context obtained.

.. code-block:: python

    async def stream_with_early_stop(
        llm: LLM,
        prompt: str,
        stop_condition
    ):
        """Stream and stop early if condition met."""
        generated_tokens = 0
        result = []

        async for chunk in llm.stream(prompt):
            result.append(chunk)
            generated_tokens += 1

            if stop_condition(result):
                # Stop generation early
                break

        return ''.join(result), generated_tokens

Batch Processing
----------------

Reduce per-request overhead.

.. code-block:: python

    class CostOptimizedBatcher:
        def __init__(self, batch_size: int = 10):
            self.batch_size = batch_size
            self.queue = []

        def add_request(self, request):
            self.queue.append(request)

            if len(self.queue) >= self.batch_size:
                return self.process_batch()

            return None

        def process_batch(self):
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]

            # Process in single API call
            results = llm.batch_generate(batch)

            # Cost savings: batch overhead < sum of individual calls
            return results

Self-Hosting
------------

Consider self-hosted models for high volume.

**Break-Even Analysis:**

.. code-block:: python

    def breakeven_analysis(
        requests_per_day: int,
        tokens_per_request: int,
        api_cost_per_million: float,
        self_hosted_monthly_cost: float
    ) -> Dict:
        """Calculate break-even point for self-hosting."""

        # API costs
        daily_api_cost = (
            requests_per_day *
            tokens_per_request /
            1_000_000 *
            api_cost_per_million
        )
        monthly_api_cost = daily_api_cost * 30

        # Self-hosted
        monthly_self_hosted = self_hosted_monthly_cost

        breakeven_months = (
            monthly_self_hosted / monthly_api_cost
            if monthly_api_cost > 0 else float('inf')
        )

        return {
            "monthly_api_cost": monthly_api_cost,
            "monthly_self_hosted_cost": monthly_self_hosted,
            "breakeven_months": breakeven_months,
            "recommendation": (
                "self-hosted" if breakeven_months < 3
                else "API"
            )
        }

    # Example
    result = breakeven_analysis(
        requests_per_day=10000,
        tokens_per_request=1000,
        api_cost_per_million=10.0,
        self_hosted_monthly_cost=2000.0
    )

Cost Budgets & Limits
=====================

Per-User Limits
---------------

.. code-block:: python

    class UserCostLimiter:
        def __init__(self, redis_client):
            self.redis = redis_client

        async def check_budget(
            self,
            user_id: str,
            estimated_cost: float,
            daily_limit: float = 1.0
        ) -> bool:
            """Check if user is within budget."""
            key = f"cost:daily:{user_id}:{date.today()}"

            # Get current usage
            current = self.redis.get(key)
            current_cost = float(current) if current else 0.0

            # Check if within limit
            if current_cost + estimated_cost > daily_limit:
                return False

            return True

        async def record_cost(
            self,
            user_id: str,
            cost: float
        ):
            """Record cost for user."""
            key = f"cost:daily:{user_id}:{date.today()}"

            # Increment with expiry
            pipe = self.redis.pipeline()
            pipe.incrbyfloat(key, cost)
            pipe.expire(key, 86400)  # 24 hours
            pipe.execute()

Organizational Limits
---------------------

.. code-block:: python

    class OrgCostManager:
        def __init__(self):
            self.org_budgets = {}
            self.org_spending = defaultdict(float)

        def set_budget(self, org_id: str, monthly_budget: float):
            self.org_budgets[org_id] = monthly_budget

        def can_process(self, org_id: str, estimated_cost: float) -> bool:
            budget = self.org_budgets.get(org_id, float('inf'))
            current = self.org_spending[org_id]

            return current + estimated_cost <= budget

        def record_spending(self, org_id: str, cost: float):
            self.org_spending[org_id] += cost

Cost Alerts
-----------

.. code-block:: python

    class CostAlerting:
        def __init__(self):
            self.thresholds = {
                "warning": 0.8,  # 80% of budget
                "critical": 0.95  # 95% of budget
            }

        def check_alerts(
            self,
            user_id: str,
            current_cost: float,
            budget: float
        ):
            usage_ratio = current_cost / budget

            if usage_ratio >= self.thresholds["critical"]:
                self.send_alert(
                    user_id,
                    f"CRITICAL: 95% of budget used (${current_cost:.2f}/${budget:.2f})"
                )
            elif usage_ratio >= self.thresholds["warning"]:
                self.send_alert(
                    user_id,
                    f"WARNING: 80% of budget used (${current_cost:.2f}/${budget:.2f})"
                )

Cost-Effective Architectures
=============================

Tiered Service
--------------

Different service levels at different costs.

.. code-block:: python

    class TieredService:
        def __init__(self):
            self.tiers = {
                "free": {
                    "model": "gpt-3.5-turbo",
                    "requests_per_day": 10,
                    "max_tokens": 500
                },
                "pro": {
                    "model": "gpt-4-turbo",
                    "requests_per_day": 100,
                    "max_tokens": 2000
                },
                "enterprise": {
                    "model": "gpt-4-turbo",
                    "requests_per_day": float('inf'),
                    "max_tokens": 8000
                }
            }

        def get_service_params(self, user_tier: str) -> Dict:
            return self.tiers.get(user_tier, self.tiers["free"])

Hybrid Approach
---------------

Combine API and self-hosted models.

.. code-block:: python

    class HybridModelRouter:
        def __init__(self):
            self.self_hosted_capacity = 100  # requests/minute
            self.current_load = 0

        def route_request(self, task: str):
            # Route simple tasks to self-hosted
            if self.is_simple(task) and self.has_capacity():
                return self.self_hosted_model.generate(task)

            # Complex or overflow to API
            return self.api_model.generate(task)

        def has_capacity(self) -> bool:
            return self.current_load < self.self_hosted_capacity

Cost Reporting
==============

.. code-block:: python

    class CostReporter:
        def __init__(self, tracker: CostTracker):
            self.tracker = tracker

        def generate_report(
            self,
            start: datetime,
            end: datetime
        ) -> Dict:
            costs = [
                c for c in self.tracker.costs
                if start <= c.timestamp <= end
            ]

            total_cost = sum(c.total_cost() for c in costs)
            total_requests = len(costs)

            llm_cost = sum(c.llm_cost for c in costs)
            tool_cost = sum(
                sum(c.tool_costs.values()) for c in costs
            )

            return {
                "period": {
                    "start": start.isoformat(),
                    "end": end.isoformat()
                },
                "summary": {
                    "total_cost": total_cost,
                    "total_requests": total_requests,
                    "cost_per_request": total_cost / total_requests if total_requests > 0 else 0
                },
                "breakdown": {
                    "llm": llm_cost,
                    "tools": tool_cost,
                    "infrastructure": total_cost - llm_cost - tool_cost
                },
                "by_user": self.tracker.cost_by_user()
            }

        def export_csv(self, filepath: str):
            """Export costs to CSV for analysis."""
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'request_id', 'timestamp', 'user_id',
                    'llm_cost', 'tool_cost', 'total_cost'
                ])
                writer.writeheader()

                for cost in self.tracker.costs:
                    writer.writerow({
                        'request_id': cost.request_id,
                        'timestamp': cost.timestamp.isoformat(),
                        'user_id': cost.user_id,
                        'llm_cost': cost.llm_cost,
                        'tool_cost': sum(cost.tool_costs.values()),
                        'total_cost': cost.total_cost()
                    })

Best Practices
==============

1. **Track Everything:** Measure all cost components
2. **Set Budgets:** User and org limits
3. **Optimize Prompts:** Reduce token usage
4. **Cache Aggressively:** Avoid redundant calls
5. **Choose Right Model:** Match model to task
6. **Monitor Usage:** Real-time cost tracking
7. **Alert on Anomalies:** Detect unusual spending
8. **Regular Reviews:** Analyze and optimize
9. **Consider Self-Hosting:** For high volume
10. **User Education:** Help users use efficiently

Resources
=========

Cost Calculators
----------------

* OpenAI Pricing Calculator
* Cloud cost estimators
* Custom cost modeling tools

See Also
========

* :doc:`accuracy`
* :doc:`latency`
* :doc:`../deployments`
* :doc:`../evaluations`
