====================
Accuracy
====================

Overview
========

Accuracy measures how well coding agents produce correct solutions. This section covers accuracy metrics, measurement methodologies, and improvement strategies.

What is Accuracy?
=================

Definition
----------

Accuracy in coding agents refers to:

* **Correctness:** Solutions pass test cases
* **Completeness:** All requirements are met
* **Consistency:** Reliable performance across tasks
* **Robustness:** Handles edge cases properly

Why Accuracy Matters
--------------------

* User trust and adoption
* Production readiness
* Safety and reliability
* Reduced manual review
* Cost efficiency (fewer retries)

Accuracy Metrics
================

Pass@k
------

Probability of generating at least one correct solution in k attempts.

**Formula:**

.. math::

    \text{Pass@k} = \mathbb{E}_{\text{Problems}} \left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]

where:
- n = total samples generated
- c = correct samples
- k = number of attempts allowed

**Implementation:**

.. code-block:: python

    import math
    from typing import List

    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Compute pass@k metric.

        Args:
            n: Total number of samples
            c: Number of correct samples
            k: Number of attempts
        """
        if n - c < k:
            return 1.0

        numerator = math.comb(n - c, k)
        denominator = math.comb(n, k)

        return 1.0 - (numerator / denominator)

**Common Values:**

* **Pass@1:** Single-shot success rate
* **Pass@10:** Success within 10 attempts
* **Pass@100:** Upper bound estimate

**Interpretation:**

.. code-block:: text

    Pass@1 = 0.65  →  65% success on first try
    Pass@10 = 0.85 →  85% success within 10 tries

    Higher values indicate better performance

Functional Correctness Rate
----------------------------

Percentage of tasks where solution is functionally correct.

.. code-block:: python

    def functional_correctness(results: List[TestResult]) -> float:
        """Calculate functional correctness rate."""
        correct = sum(1 for r in results if r.all_tests_passed)
        return correct / len(results) if results else 0.0

Test Case Pass Rate
-------------------

Average percentage of test cases passed per problem.

.. code-block:: python

    def test_case_pass_rate(results: List[TestResult]) -> float:
        """Calculate average test case pass rate."""
        total_tests = 0
        passed_tests = 0

        for result in results:
            total_tests += result.total_tests
            passed_tests += result.passed_tests

        return passed_tests / total_tests if total_tests > 0 else 0.0

Partial Correctness
-------------------

Credit for partially correct solutions.

.. code-block:: python

    @dataclass
    class PartialCorrectnessScore:
        syntax_correct: bool  # 0.2
        compiles: bool  # 0.2
        passes_basic_tests: float  # 0.3
        passes_edge_cases: float  # 0.3

        def compute_score(self) -> float:
            score = 0.0
            if self.syntax_correct:
                score += 0.2
            if self.compiles:
                score += 0.2
            score += 0.3 * self.passes_basic_tests
            score += 0.3 * self.passes_edge_cases
            return score

Exact Match
-----------

Solution exactly matches expected output.

.. code-block:: python

    def exact_match_rate(
        generated: List[str],
        expected: List[str]
    ) -> float:
        """Calculate exact match rate."""
        matches = sum(
            1 for g, e in zip(generated, expected)
            if normalize(g) == normalize(e)
        )
        return matches / len(generated)

Semantic Equivalence
--------------------

Solutions are semantically equivalent though syntactically different.

.. code-block:: python

    def semantic_equivalence(
        solution1: str,
        solution2: str,
        test_suite: List[Test]
    ) -> bool:
        """Check if two solutions are semantically equivalent."""
        for test in test_suite:
            result1 = execute_with_test(solution1, test)
            result2 = execute_with_test(solution2, test)

            if result1 != result2:
                return False

        return True

Measurement Methodologies
==========================

Unit Test Execution
-------------------

Run generated code against test suites.

**Process:**

1. Generate code solution
2. Execute with test inputs
3. Compare outputs with expected
4. Aggregate results

**Example:**

.. code-block:: python

    class TestExecutor:
        def __init__(self, timeout: int = 5):
            self.timeout = timeout

        def execute_test(
            self,
            code: str,
            test: Test
        ) -> TestResult:
            """Execute code with a single test."""
            try:
                # Setup environment
                exec_globals = {}
                exec(code, exec_globals)

                # Get function
                func = exec_globals[test.function_name]

                # Run with timeout
                result = self._run_with_timeout(
                    func,
                    test.input,
                    self.timeout
                )

                # Check correctness
                passed = result == test.expected_output

                return TestResult(
                    passed=passed,
                    actual=result,
                    expected=test.expected_output
                )

            except Exception as e:
                return TestResult(
                    passed=False,
                    error=str(e)
                )

Property-Based Testing
----------------------

Generate random test cases checking properties.

.. code-block:: python

    from hypothesis import given, strategies as st

    def test_sorting_properties(sorting_function):
        """Test sorting function properties."""

        @given(st.lists(st.integers()))
        def test_sorted_output_is_sorted(arr):
            result = sorting_function(arr)
            assert all(result[i] <= result[i+1]
                      for i in range(len(result)-1))

        @given(st.lists(st.integers()))
        def test_same_length(arr):
            result = sorting_function(arr)
            assert len(result) == len(arr)

        @given(st.lists(st.integers()))
        def test_same_elements(arr):
            result = sorting_function(arr)
            assert sorted(arr) == result

        # Run tests
        test_sorted_output_is_sorted()
        test_same_length()
        test_same_elements()

Mutation Testing
----------------

Introduce bugs to verify test effectiveness.

.. code-block:: python

    class MutationTester:
        def __init__(self):
            self.mutators = [
                self.swap_operators,
                self.change_constants,
                self.modify_conditions
            ]

        def test_robustness(
            self,
            code: str,
            tests: List[Test]
        ) -> float:
            """Test how well code handles mutations."""
            killed_mutants = 0
            total_mutants = 0

            for mutator in self.mutators:
                mutated_codes = mutator(code)

                for mutated in mutated_codes:
                    total_mutants += 1

                    # Check if tests catch mutation
                    for test in tests:
                        result = execute_with_test(mutated, test)
                        if not result.passed:
                            killed_mutants += 1
                            break

            # Mutation score
            return killed_mutants / total_mutants if total_mutants > 0 else 0.0

Human Evaluation
----------------

Expert review of correctness and quality.

.. code-block:: python

    @dataclass
    class HumanEvaluationScore:
        correctness: int  # 1-5
        completeness: int  # 1-5
        edge_case_handling: int  # 1-5
        code_quality: int  # 1-5
        comments: str

        def accuracy_score(self) -> float:
            """Compute accuracy component of score."""
            return (
                self.correctness * 0.4 +
                self.completeness * 0.3 +
                self.edge_case_handling * 0.3
            ) / 5.0

Accuracy by Task Type
=====================

Simple Functions
----------------

Single-purpose functions with clear specifications.

**Typical Accuracy:** 70-85% Pass@1

**Characteristics:**

* Well-defined inputs/outputs
* Limited complexity
* Few edge cases
* Standard algorithms

**Example Tasks:**

* String manipulation
* List operations
* Basic math functions
* Simple parsers

Complex Algorithms
------------------

Multi-step algorithmic problems.

**Typical Accuracy:** 30-50% Pass@1

**Characteristics:**

* Multiple steps
* Complex logic
* Many edge cases
* Performance requirements

**Example Tasks:**

* Graph algorithms
* Dynamic programming
* Optimization problems
* Complex data structures

Repository-Level Tasks
----------------------

Multi-file changes in real codebases.

**Typical Accuracy:** 10-30% full resolution

**Characteristics:**

* Large context
* Multiple files
* Complex dependencies
* Real-world constraints

**Example Tasks:**

* Bug fixes in open source
* Feature additions
* Refactoring
* Integration tasks

Improving Accuracy
==================

Prompt Engineering
------------------

Better prompts lead to more accurate outputs.

**Techniques:**

Clear Instructions
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Bad:  Write a function
    Good: Write a Python function that takes a list of integers
          and returns the sum of even numbers. Handle empty lists.

Examples
~~~~~~~~

.. code-block:: text

    Write a function to calculate factorial.

    Example:
    Input: 5
    Output: 120

    Handle edge cases:
    - factorial(0) should return 1
    - factorial(1) should return 1
    - Negative inputs should raise ValueError

Chain-of-Thought
~~~~~~~~~~~~~~~~

.. code-block:: text

    Before writing code, think through:
    1. What are the inputs and outputs?
    2. What are the edge cases?
    3. What algorithm is appropriate?
    4. How can I verify correctness?

Test-Driven Prompting
---------------------

Include tests in prompt.

.. code-block:: text

    Write a function that passes these tests:

    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True
    assert is_palindrome("a") == True

Self-Verification
-----------------

Agent checks its own solutions.

.. code-block:: python

    async def generate_with_verification(
        task: str,
        tests: List[Test]
    ) -> str:
        """Generate code with self-verification."""
        max_attempts = 3

        for attempt in range(max_attempts):
            # Generate solution
            code = await agent.generate_code(task)

            # Self-verify
            results = execute_tests(code, tests)

            if all(r.passed for r in results):
                return code

            # If failed, provide feedback for next attempt
            feedback = generate_feedback(results)
            task = f"{task}\n\nPrevious attempt failed:\n{feedback}"

        return code  # Return best attempt

Iterative Refinement
--------------------

Improve solutions through iterations.

.. code-block:: python

    async def iterative_refinement(
        initial_solution: str,
        tests: List[Test],
        max_iterations: int = 3
    ) -> str:
        """Refine solution iteratively."""
        solution = initial_solution

        for i in range(max_iterations):
            # Test current solution
            results = execute_tests(solution, tests)

            if all(r.passed for r in results):
                return solution

            # Identify failures
            failures = [r for r in results if not r.passed]

            # Generate fix
            fix_prompt = f"""
            Current code:
            {solution}

            Failed tests:
            {format_failures(failures)}

            Fix the code to pass all tests.
            """

            solution = await agent.generate_code(fix_prompt)

        return solution

Ensemble Methods
----------------

Combine multiple solutions.

.. code-block:: python

    async def ensemble_generation(
        task: str,
        n_samples: int = 5,
        tests: List[Test] = None
    ) -> str:
        """Generate multiple solutions and select best."""
        solutions = []

        # Generate multiple solutions
        for _ in range(n_samples):
            solution = await agent.generate_code(task)
            solutions.append(solution)

        # If tests available, select best
        if tests:
            scored = []
            for solution in solutions:
                results = execute_tests(solution, tests)
                score = sum(1 for r in results if r.passed) / len(results)
                scored.append((solution, score))

            # Return best scoring
            return max(scored, key=lambda x: x[1])[0]

        # Otherwise, use voting or first valid
        return solutions[0]

Fine-Tuning
-----------

Train on correct examples.

**Data:**

.. code-block:: python

    training_examples = [
        {
            "instruction": "Write a function to reverse a string",
            "input": "def reverse_string(s: str) -> str:",
            "output": "    return s[::-1]"
        },
        # ... more examples
    ]

**Benefits:**

* Higher accuracy on similar tasks
* Better pattern recognition
* Improved edge case handling
* Domain adaptation

Accuracy Monitoring
===================

Continuous Evaluation
---------------------

Regular accuracy assessment.

.. code-block:: python

    class AccuracyMonitor:
        def __init__(self):
            self.results = []

        def log_result(self, task_id: str, passed: bool, timestamp: datetime):
            self.results.append({
                "task_id": task_id,
                "passed": passed,
                "timestamp": timestamp
            })

        def current_accuracy(self, window: timedelta = timedelta(hours=1)) -> float:
            """Calculate accuracy in recent window."""
            cutoff = datetime.now() - window
            recent = [r for r in self.results if r["timestamp"] >= cutoff]

            if not recent:
                return 0.0

            passed = sum(1 for r in recent if r["passed"])
            return passed / len(recent)

        def accuracy_trend(self) -> List[Tuple[datetime, float]]:
            """Calculate accuracy over time."""
            # Group by hour
            hourly = defaultdict(list)
            for r in self.results:
                hour = r["timestamp"].replace(minute=0, second=0, microsecond=0)
                hourly[hour].append(r["passed"])

            # Calculate accuracy per hour
            trend = []
            for hour in sorted(hourly.keys()):
                passed = sum(hourly[hour])
                total = len(hourly[hour])
                trend.append((hour, passed / total))

            return trend

Error Analysis
--------------

Understand accuracy failures.

.. code-block:: python

    @dataclass
    class ErrorAnalysis:
        syntax_errors: int
        runtime_errors: int
        logic_errors: int
        edge_case_failures: int
        timeout_errors: int

        def most_common_error(self) -> str:
            errors = {
                "syntax": self.syntax_errors,
                "runtime": self.runtime_errors,
                "logic": self.logic_errors,
                "edge_case": self.edge_case_failures,
                "timeout": self.timeout_errors
            }
            return max(errors.items(), key=lambda x: x[1])[0]

        def total_errors(self) -> int:
            return sum([
                self.syntax_errors,
                self.runtime_errors,
                self.logic_errors,
                self.edge_case_failures,
                self.timeout_errors
            ])

Accuracy vs. Other Metrics
===========================

Accuracy vs. Latency
--------------------

Trade-off between accuracy and response time.

**Strategies:**

* Fast first-pass, iterate if needed
* Complexity-based timeout
* User-configurable trade-off

Accuracy vs. Cost
-----------------

More accurate solutions often cost more.

**Optimization:**

* Use cheaper models for simple tasks
* Ensemble only when needed
* Cache high-confidence results
* Early stopping on high-quality solutions

Accuracy vs. Coverage
---------------------

Balance between solving problems correctly vs. attempting all problems.

**Decision:**

.. code-block:: python

    def should_attempt(task: Task, confidence_threshold: float = 0.7) -> bool:
        """Decide whether to attempt task based on confidence."""
        confidence = estimate_success_probability(task)
        return confidence >= confidence_threshold

Best Practices
==============

1. **Comprehensive Testing:** Use diverse test suites
2. **Monitor Continuously:** Track accuracy over time
3. **Analyze Failures:** Understand why solutions fail
4. **Iterate:** Continuous improvement
5. **Balance Trade-offs:** Consider accuracy vs. other metrics
6. **Version Control:** Track accuracy by model version
7. **A/B Test:** Compare accuracy of different approaches
8. **User Feedback:** Incorporate real-world accuracy data
9. **Edge Cases:** Pay special attention to edge case handling
10. **Graceful Degradation:** Partial solutions better than nothing

Resources
=========

Benchmarks
----------

* HumanEval: Standard function-level benchmark
* MBPP: Basic programming problems
* APPS: Algorithmic problems
* SWE-bench: Repository-level tasks

Tools
-----

* pytest: Testing framework
* Hypothesis: Property-based testing
* Mutation testing tools
* Code coverage tools

See Also
========

* :doc:`../llm/benchmarking`
* :doc:`../evaluations`
* :doc:`latency`
* :doc:`cost`
