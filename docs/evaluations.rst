====================
Agent Evaluations
====================

Overview
========

Evaluating coding agents is essential for measuring performance, identifying weaknesses, and guiding improvements. This section covers evaluation methodologies, metrics, and best practices.

Why Evaluate?
=============

Objectives
----------

* **Performance Measurement:** Quantify agent capabilities
* **Comparison:** Benchmark against baselines and competitors
* **Quality Assurance:** Ensure reliability and safety
* **Progress Tracking:** Monitor improvements over time
* **Decision Making:** Guide development priorities

Challenges
----------

* Multi-dimensional performance
* Subjective quality aspects
* Task complexity variation
* Evaluation cost
* Benchmark contamination

Evaluation Dimensions
=====================

Functional Correctness
----------------------

Does the agent produce correct solutions?

**Metrics:**

* Pass@k rate on test suites
* Unit test pass rate
* Functional equivalence
* Bug introduction rate

**Measurement:**

.. code-block:: python

    def evaluate_correctness(
        generated_code: str,
        test_suite: List[Test]
    ) -> float:
        passed = 0
        for test in test_suite:
            try:
                result = execute_with_test(generated_code, test)
                if result.passed:
                    passed += 1
            except Exception:
                pass

        return passed / len(test_suite)

Code Quality
------------

Is the generated code well-written?

**Aspects:**

* Readability
* Maintainability
* Efficiency
* Following best practices
* Documentation quality

**Metrics:**

.. code-block:: python

    @dataclass
    class CodeQualityScore:
        readability: float  # 0-1
        complexity: float   # Cyclomatic complexity
        duplication: float  # Code duplication %
        test_coverage: float
        documentation: float
        style_compliance: float

        def overall_score(self) -> float:
            return (
                self.readability * 0.2 +
                (1 - self.complexity / 20) * 0.2 +
                (1 - self.duplication) * 0.15 +
                self.test_coverage * 0.2 +
                self.documentation * 0.15 +
                self.style_compliance * 0.1
            )

**Tools:**

* Linters (pylint, eslint)
* Complexity analyzers (radon, lizard)
* Code smell detectors (SonarQube)
* Style checkers (black, prettier)

Task Completion
---------------

Can the agent complete end-to-end tasks?

**Metrics:**

* Task success rate
* Partial completion rate
* Time to completion
* Number of iterations needed

**Example Evaluation:**

.. code-block:: python

    @dataclass
    class TaskResult:
        task_id: str
        completed: bool
        partial_completion: float  # 0-1
        time_seconds: float
        iterations: int
        errors_encountered: List[str]

    def evaluate_tasks(
        agent: Agent,
        tasks: List[Task]
    ) -> TaskEvaluationReport:
        results = []

        for task in tasks:
            start_time = time.time()
            result = agent.execute(task)
            duration = time.time() - start_time

            results.append(TaskResult(
                task_id=task.id,
                completed=result.success,
                partial_completion=result.completion_ratio,
                time_seconds=duration,
                iterations=result.iterations,
                errors_encountered=result.errors
            ))

        return TaskEvaluationReport(results)

User Interaction Quality
-------------------------

How well does the agent communicate?

**Aspects:**

* Clarity of explanations
* Appropriate level of detail
* Proactive communication
* Error handling
* Question asking

**Evaluation:**

.. code-block:: python

    class InteractionEvaluator:
        def evaluate_response(
            self,
            response: str,
            context: Dict
        ) -> InteractionScore:
            return InteractionScore(
                clarity=self.score_clarity(response),
                completeness=self.score_completeness(response, context),
                tone=self.score_tone(response),
                helpfulness=self.score_helpfulness(response)
            )

Security
--------

Does the agent generate secure code?

**Checks:**

* Common vulnerabilities (OWASP Top 10)
* Input validation
* Authentication/authorization
* Secrets exposure
* Injection vulnerabilities

**Tools:**

* Bandit (Python)
* ESLint security plugins
* Semgrep
* CodeQL

**Example:**

.. code-block:: python

    def evaluate_security(code: str, language: str) -> SecurityReport:
        vulnerabilities = []

        # Run security scanners
        if language == "python":
            bandit_results = run_bandit(code)
            vulnerabilities.extend(bandit_results)

        semgrep_results = run_semgrep(code, language)
        vulnerabilities.extend(semgrep_results)

        return SecurityReport(
            vulnerability_count=len(vulnerabilities),
            critical_count=sum(1 for v in vulnerabilities if v.severity == "critical"),
            vulnerabilities=vulnerabilities
        )

Efficiency
----------

Resource usage and performance.

**Metrics:**

* Inference time / latency
* Token usage
* API call count
* Memory consumption
* Cost per task

Tool Use Effectiveness
----------------------

How well does the agent use tools?

**Metrics:**

* Tool selection accuracy
* Successful tool invocations
* Unnecessary tool calls
* Error recovery

Evaluation Methodologies
=========================

Automated Testing
-----------------

Run agents on test suites automatically.

HumanEval/MBPP Style
~~~~~~~~~~~~~~~~~~~~

Function-level code generation with unit tests.

.. code-block:: python

    def evaluate_humaneval(agent: Agent, dataset: List[Problem]):
        results = []

        for problem in dataset:
            # Generate solution
            solution = agent.generate_solution(problem.prompt)

            # Run tests
            test_results = run_tests(solution, problem.tests)

            results.append({
                "problem_id": problem.id,
                "passed": test_results.all_passed(),
                "pass_rate": test_results.pass_rate()
            })

        # Calculate Pass@k
        pass_at_1 = sum(r["passed"] for r in results) / len(results)
        return {"pass@1": pass_at_1}

Repository-Level Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full repository tasks like SWE-bench.

.. code-block:: python

    def evaluate_swe_bench(
        agent: Agent,
        issues: List[GitHubIssue]
    ):
        results = []

        for issue in issues:
            # Setup repository
            repo = clone_repository(issue.repo_url, issue.base_commit)

            # Agent attempts to resolve
            solution = agent.resolve_issue(
                issue.description,
                repo
            )

            # Apply changes and run tests
            apply_changes(repo, solution.changes)
            test_results = run_test_suite(repo)

            results.append({
                "issue_id": issue.id,
                "resolved": test_results.passed and passes_gold_tests(issue),
                "files_changed": len(solution.changes)
            })

        return compute_resolution_rate(results)

Interactive Evaluation
----------------------

Evaluate multi-turn interactions.

.. code-block:: python

    def evaluate_interactive_session(
        agent: Agent,
        scenario: InteractiveScenario
    ):
        conversation = []
        state = scenario.initial_state

        for turn in scenario.turns:
            # User message
            conversation.append({"role": "user", "content": turn.user_message})

            # Agent response
            response = agent.respond(conversation, state)
            conversation.append({"role": "assistant", "content": response})

            # Update state
            state = scenario.update_state(state, response)

            # Evaluate turn
            turn_score = evaluate_turn(
                response,
                turn.expected_behavior,
                state
            )

        # Overall evaluation
        return InteractiveEvaluationResult(
            turns=turn_scores,
            final_state_correct=scenario.verify_final_state(state),
            conversation_quality=evaluate_conversation(conversation)
        )

Human Evaluation
----------------

Expert review of agent outputs.

**Process:**

1. **Setup:** Define evaluation criteria
2. **Sampling:** Select representative examples
3. **Review:** Experts assess outputs
4. **Scoring:** Apply rubrics
5. **Analysis:** Aggregate and analyze

**Rubric Example:**

.. code-block:: python

    @dataclass
    class HumanEvaluationRubric:
        correctness: int  # 1-5
        code_quality: int  # 1-5
        explanation_quality: int  # 1-5
        appropriateness: int  # 1-5
        comments: str

    def human_evaluation(
        samples: List[AgentOutput],
        evaluators: List[HumanEvaluator]
    ):
        evaluations = []

        for sample in samples:
            sample_evals = []
            for evaluator in evaluators:
                score = evaluator.evaluate(sample)
                sample_evals.append(score)

            # Calculate inter-rater reliability
            agreement = calculate_agreement(sample_evals)

            evaluations.append({
                "sample_id": sample.id,
                "scores": sample_evals,
                "mean_scores": compute_means(sample_evals),
                "agreement": agreement
            })

        return evaluations

A/B Testing
-----------

Compare agent versions in production.

.. code-block:: python

    class ABTest:
        def __init__(self, agent_a: Agent, agent_b: Agent):
            self.agent_a = agent_a
            self.agent_b = agent_b
            self.results = {"A": [], "B": []}

        def run_test(self, tasks: List[Task]):
            for task in tasks:
                # Randomly assign
                variant = random.choice(["A", "B"])
                agent = self.agent_a if variant == "A" else self.agent_b

                # Execute and measure
                start = time.time()
                result = agent.execute(task)
                duration = time.time() - start

                self.results[variant].append({
                    "success": result.success,
                    "duration": duration,
                    "quality": evaluate_quality(result)
                })

        def analyze(self):
            # Statistical analysis
            a_success = success_rate(self.results["A"])
            b_success = success_rate(self.results["B"])

            p_value = statistical_test(
                self.results["A"],
                self.results["B"]
            )

            return ABTestResult(
                variant_a_success=a_success,
                variant_b_success=b_success,
                statistical_significance=p_value < 0.05,
                p_value=p_value
            )

Benchmark Suites
================

Popular Benchmarks
------------------

See :doc:`llm/benchmarking` for detailed benchmark descriptions.

**Code Generation:**

* HumanEval
* MBPP
* APPS
* CodeContests

**Repository-Level:**

* SWE-bench
* RepoEval
* CrossCodeEval

**Code Understanding:**

* CodeXGLUE
* CodeSearchNet

Custom Benchmarks
-----------------

Create domain-specific evaluation sets.

.. code-block:: python

    class CustomBenchmark:
        def __init__(self, name: str, tasks: List[Task]):
            self.name = name
            self.tasks = tasks

        def evaluate(self, agent: Agent) -> BenchmarkResult:
            results = []

            for task in self.tasks:
                result = agent.execute(task)

                results.append(TaskResult(
                    task_id=task.id,
                    success=self.verify_solution(task, result),
                    metrics=self.compute_metrics(task, result)
                ))

            return BenchmarkResult(
                benchmark_name=self.name,
                task_results=results,
                overall_score=self.aggregate_score(results)
            )

        def verify_solution(self, task: Task, result: Any) -> bool:
            # Custom verification logic
            pass

        def compute_metrics(self, task: Task, result: Any) -> Dict:
            # Custom metrics
            pass

Evaluation Metrics
==================

Success Metrics
---------------

Pass@k
~~~~~~

Probability of at least one correct solution in k samples.

.. code-block:: python

    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """
        n: total samples
        c: correct samples
        k: number of attempts
        """
        if n - c < k:
            return 1.0
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

Completion Rate
~~~~~~~~~~~~~~~

Percentage of tasks fully completed.

.. code-block:: python

    def completion_rate(results: List[TaskResult]) -> float:
        completed = sum(1 for r in results if r.completed)
        return completed / len(results)

Performance Metrics
-------------------

Latency
~~~~~~~

Time to generate solutions.

.. code-block:: python

    @dataclass
    class LatencyMetrics:
        mean: float
        median: float
        p95: float
        p99: float

    def compute_latency_metrics(
        durations: List[float]
    ) -> LatencyMetrics:
        return LatencyMetrics(
            mean=statistics.mean(durations),
            median=statistics.median(durations),
            p95=numpy.percentile(durations, 95),
            p99=numpy.percentile(durations, 99)
        )

Throughput
~~~~~~~~~~

Tasks completed per unit time.

.. code-block:: python

    def compute_throughput(
        num_tasks: int,
        total_time: float
    ) -> float:
        return num_tasks / total_time  # tasks per second

Cost Metrics
------------

Token Usage
~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class CostMetrics:
        total_tokens: int
        input_tokens: int
        output_tokens: int
        total_cost_usd: float
        cost_per_task: float

API Calls
~~~~~~~~~

Number and cost of external API calls.

Quality Metrics
---------------

Code Quality Score
~~~~~~~~~~~~~~~~~~

Composite metric of code quality aspects.

.. code-block:: python

    def compute_code_quality_score(code: str) -> float:
        scores = {
            "readability": analyze_readability(code),
            "complexity": analyze_complexity(code),
            "style": check_style_compliance(code),
            "documentation": check_documentation(code)
        }

        # Weighted average
        weights = {
            "readability": 0.3,
            "complexity": 0.3,
            "style": 0.2,
            "documentation": 0.2
        }

        return sum(scores[k] * weights[k] for k in scores)

Test Coverage
~~~~~~~~~~~~~

Percentage of code covered by tests.

Evaluation Tools
================

Execution Sandboxes
-------------------

Safe code execution environments.

Docker
~~~~~~

.. code-block:: python

    import docker

    def execute_in_docker(code: str, tests: str) -> ExecutionResult:
        client = docker.from_env()

        container = client.containers.run(
            "python:3.9",
            command=f"python -c '{code}; {tests}'",
            detach=True,
            mem_limit="512m",
            cpu_period=100000,
            cpu_quota=50000
        )

        # Wait and collect output
        result = container.wait()
        logs = container.logs().decode()
        container.remove()

        return ExecutionResult(
            exit_code=result["StatusCode"],
            output=logs
        )

Evaluation Frameworks
---------------------

BigCode Evaluation Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified evaluation for code models.

.. code-block:: bash

    # Install
    pip install bigcode-evaluation-harness

    # Run evaluation
    bigcode-eval \
        --model codegen-350M \
        --tasks humaneval \
        --n_samples 10 \
        --batch_size 1

EvalPlus
~~~~~~~~

Enhanced evaluation with more tests.

.. code-block:: python

    from evalplus.evaluate import evaluate

    results = evaluate(
        dataset="humaneval+",
        samples=generated_samples,
        parallel=4
    )

Custom Evaluation Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class EvaluationSuite:
        def __init__(self, name: str):
            self.name = name
            self.evaluators = []

        def add_evaluator(self, evaluator: Evaluator):
            self.evaluators.append(evaluator)

        def run(self, agent: Agent, dataset: Dataset) -> Report:
            results = {}

            for evaluator in self.evaluators:
                results[evaluator.name] = evaluator.evaluate(
                    agent,
                    dataset
                )

            return Report(
                suite_name=self.name,
                results=results,
                timestamp=datetime.now()
            )

Analysis & Reporting
====================

Statistical Analysis
--------------------

.. code-block:: python

    from scipy import stats

    def compare_agents(
        agent_a_results: List[float],
        agent_b_results: List[float]
    ) -> ComparisonResult:
        # T-test
        t_stat, p_value = stats.ttest_ind(
            agent_a_results,
            agent_b_results
        )

        # Effect size (Cohen's d)
        mean_a = np.mean(agent_a_results)
        mean_b = np.mean(agent_b_results)
        pooled_std = np.sqrt(
            (np.std(agent_a_results)**2 + np.std(agent_b_results)**2) / 2
        )
        cohens_d = (mean_a - mean_b) / pooled_std

        return ComparisonResult(
            significant=p_value < 0.05,
            p_value=p_value,
            effect_size=cohens_d,
            winner="A" if mean_a > mean_b else "B"
        )

Visualization
-------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_evaluation_results(results: Dict[str, List[float]]):
        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=results)
        plt.title("Agent Performance Comparison")
        plt.ylabel("Success Rate")
        plt.savefig("evaluation_results.png")

        # Performance over time
        plt.figure(figsize=(12, 6))
        for agent, scores in results.items():
            plt.plot(scores, label=agent)
        plt.xlabel("Evaluation Round")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("performance_trend.png")

Reporting
---------

.. code-block:: python

    @dataclass
    class EvaluationReport:
        agent_name: str
        timestamp: datetime
        metrics: Dict[str, float]
        task_results: List[TaskResult]
        summary: str

        def to_markdown(self) -> str:
            return f"""
# Evaluation Report: {self.agent_name}

**Date:** {self.timestamp}

## Summary
{self.summary}

## Metrics
{self._format_metrics()}

## Detailed Results
{self._format_task_results()}
            """

        def to_json(self) -> str:
            return json.dumps(dataclasses.asdict(self), default=str)

Best Practices
==============

Evaluation Design
-----------------

1. **Diverse Tasks:** Cover different difficulty levels and domains
2. **Reproducible:** Fix random seeds, version dependencies
3. **Realistic:** Use real-world scenarios
4. **Multiple Metrics:** Don't rely on single metric
5. **Regular Cadence:** Evaluate frequently

Implementation
--------------

1. **Automate:** Automated evaluation pipelines
2. **Isolate:** Sandboxed execution
3. **Monitor:** Track evaluation costs
4. **Version Control:** Version datasets and evaluation code
5. **Document:** Clear documentation of methodology

Analysis
--------

1. **Statistical Rigor:** Use proper statistical tests
2. **Error Analysis:** Analyze failure cases
3. **Ablation Studies:** Understand component contributions
4. **Qualitative Review:** Manual inspection of samples
5. **Compare Fairly:** Same conditions for all agents

Common Pitfalls
===============

* **Data Contamination:** Training on eval data
* **Metric Gaming:** Optimizing for metrics not true quality
* **Insufficient Coverage:** Limited test cases
* **Ignoring Errors:** Not analyzing failures
* **Cost Neglect:** Not considering inference cost

Resources
=========

Benchmarks
----------

* HumanEval: https://github.com/openai/human-eval
* MBPP: https://github.com/google-research/google-research/tree/master/mbpp
* SWE-bench: https://www.swebench.com
* EvalPlus: https://evalplus.github.io

Tools
-----

* BigCode Evaluation Harness
* EvalPlus
* Docker (sandboxing)
* pytest, unittest (testing)

Papers
------

[Add relevant papers on evaluation methodologies]

See Also
========

* :doc:`llm/benchmarking`
* :doc:`performance/accuracy`
* :doc:`deployments`
