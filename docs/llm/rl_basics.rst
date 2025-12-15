====================
Reinforcement Learning Basics
====================

Overview
========

Reinforcement Learning (RL) is increasingly used to improve coding agents through trial-and-error learning and reward optimization. This section covers RL fundamentals relevant to coding agent development.

RL Fundamentals
===============

Core Concepts
-------------

**Agent**
  The decision-maker (e.g., coding agent, LLM)

**Environment**
  The world the agent interacts with (e.g., codebase, IDE, compiler)

**State (s)**
  Current situation/observation (e.g., code context, error messages)

**Action (a)**
  Agent's choice (e.g., generate code, use tool, ask question)

**Reward (r)**
  Feedback signal (e.g., +1 for passing tests, -1 for errors)

**Policy (π)**
  Agent's strategy: π(a|s) - probability of action a in state s

**Value Function (V)**
  Expected cumulative reward from a state

**Q-Function (Q)**
  Expected cumulative reward from taking action a in state s

The RL Loop
-----------

.. code-block:: text

    1. Agent observes state s_t
    2. Agent selects action a_t based on policy π
    3. Environment transitions to s_{t+1}
    4. Agent receives reward r_t
    5. Agent updates policy
    6. Repeat

**Goal:** Maximize cumulative reward over time

.. math::

    \max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]

where γ is the discount factor (0 < γ ≤ 1).

Key RL Concepts
===============

Exploration vs. Exploitation
-----------------------------

**Exploration:** Try new actions to discover better strategies

**Exploitation:** Use known good actions to maximize reward

**Trade-off:**
  Too much exploration → inefficient learning
  Too much exploitation → stuck in local optima

**Strategies:**

* ε-greedy: Random action with probability ε
* Softmax: Sample proportional to Q-values
* UCB: Upper confidence bound
* Optimistic initialization

Credit Assignment
-----------------

Problem: Which actions led to which rewards?

**Challenges:**

* Delayed rewards
* Long action sequences
* Sparse feedback

**Solutions:**

* Discount factor (γ)
* Eligibility traces
* Advantage estimation

On-Policy vs. Off-Policy
-------------------------

**On-Policy:**
  Learn about the policy being followed (e.g., SARSA, PPO)

**Off-Policy:**
  Learn about one policy while following another (e.g., Q-learning, DQN)

**Trade-offs:**

* On-policy: More stable, less sample efficient
* Off-policy: More sample efficient, less stable

RL Algorithms
=============

Value-Based Methods
-------------------

Q-Learning
~~~~~~~~~~

Learn Q-values directly.

**Update Rule:**

.. math::

    Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]

**Characteristics:**

* Off-policy
* Model-free
* Simple but effective

Deep Q-Networks (DQN)
~~~~~~~~~~~~~~~~~~~~~

Q-learning with neural networks.

**Innovations:**

* Experience replay
* Target network
* Double DQN improvements

Policy-Based Methods
--------------------

REINFORCE
~~~~~~~~~

Direct policy optimization using gradient ascent.

**Update:**

.. math::

    \nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]

**Characteristics:**

* On-policy
* High variance
* Simple gradient-based

Policy Gradient Theorem
~~~~~~~~~~~~~~~~~~~~~~~

Foundation for policy optimization:

.. math::

    \nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]

Actor-Critic Methods
--------------------

Combine policy-based (actor) and value-based (critic).

**Components:**

* **Actor:** Updates policy
* **Critic:** Estimates value function

**Advantages:**

* Lower variance than pure policy gradient
* More efficient than pure value-based

A2C (Advantage Actor-Critic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses advantage function: A(s,a) = Q(s,a) - V(s)

**Benefits:**

* Reduces variance
* Stabilizes training
* Better gradient estimates

A3C (Asynchronous A2C)
~~~~~~~~~~~~~~~~~~~~~~

Parallel training with multiple agents.

**Key Idea:**

* Multiple workers explore simultaneously
* Share gradients asynchronously
* More diverse experience

Proximal Policy Optimization (PPO)
-----------------------------------

State-of-the-art policy optimization algorithm.

**Key Innovation:**

Clipped objective prevents large policy updates:

.. math::

    L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

**Advantages:**

* Simple to implement
* Sample efficient
* Stable training
* Good performance

**Why PPO for LLMs:**

* Handles high-dimensional action spaces
* Robust to hyperparameters
* Scales to large models

RL for Language Models
======================

RLHF Pipeline
-------------

See :doc:`post_training` for detailed RLHF coverage.

**Key Steps:**

1. Supervised fine-tuning (SFT)
2. Reward model training
3. RL optimization (typically PPO)
4. Evaluation and iteration

Reward Models
-------------

**Purpose:** Predict human preferences

**Training:**

* Collect comparison data (A vs B)
* Train classifier/ranker
* Use as reward signal for RL

**Architecture:**

* Based on language model
* Classification head for preferences
* Typically transformer-based

**Challenges:**

* Reward hacking
* Distributional shift
* Scalability

KL Divergence Constraint
------------------------

Prevent policy from deviating too much from reference:

.. math::

    R_{total} = R_{task} - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})

**Purpose:**

* Maintain general capabilities
* Prevent reward over-optimization
* Stabilize training

RL for Coding Agents
====================

State Representation
--------------------

**Components:**

* Code context
* File structure
* Error messages
* Test results
* Tool outputs
* Conversation history

**Challenges:**

* High-dimensional
* Variable length
* Mixed modalities

Action Space
------------

**Possible Actions:**

* Generate code snippet
* Edit specific lines
* Use tools (search, compile, test)
* Ask clarifying questions
* Request more context

**Challenges:**

* Extremely large action space
* Structured outputs
* Syntactic constraints
* Semantic validity

Reward Design
-------------

**Reward Sources:**

Unit Tests
~~~~~~~~~~

* +1 for passing test
* -1 for failing test
* Partial credit for subset

Code Quality
~~~~~~~~~~~~

* Readability metrics
* Complexity scores
* Style compliance
* Security analysis

Efficiency
~~~~~~~~~~

* Runtime performance
* Memory usage
* Token efficiency
* Inference cost

Task Completion
~~~~~~~~~~~~~~~

* Solved: +10
* Partial: +5
* Failed: -5
* Timeout: -2

**Reward Shaping:**

* Intermediate rewards for progress
* Penalties for invalid code
* Bonuses for elegant solutions
* Costs for expensive operations

RL Training Strategies
----------------------

Curriculum Learning
~~~~~~~~~~~~~~~~~~~

Start with easy problems, gradually increase difficulty.

**Benefits:**

* Faster initial learning
* More stable training
* Better final performance

Self-Play
~~~~~~~~~

Agent generates training data through interaction.

**Applications:**

* Code generation and testing
* Review and refinement
* Multi-agent collaboration

Imitation Learning
~~~~~~~~~~~~~~~~~~

Learn from expert demonstrations before RL.

**Approaches:**

* Behavioral cloning
* DAgger (Dataset Aggregation)
* GAIL (Generative Adversarial Imitation)

Practical Considerations
========================

Sample Efficiency
-----------------

**Challenge:** RL requires many samples

**Solutions:**

* Transfer learning from SFT
* Off-policy methods
* Experience replay
* Synthetic environments

Stability
---------

**Issues:**

* Training instability
* Catastrophic forgetting
* Reward hacking

**Mitigations:**

* PPO clipping
* Regularization
* KL constraints
* Checkpointing

Scalability
-----------

**Challenges:**

* Large model size
* Expensive inference
* Memory requirements

**Solutions:**

* LoRA for efficient training
* Gradient checkpointing
* Model parallelism
* Distillation

Evaluation
----------

**Metrics:**

* Task success rate
* Reward curves
* Policy entropy
* KL divergence from reference

**Validation:**

* Hold-out test set
* Human evaluation
* Real-world deployment
* A/B testing

Advanced Topics
===============

Offline RL
----------

Learn from fixed datasets without environment interaction.

**Algorithms:**

* Conservative Q-Learning (CQL)
* Implicit Q-Learning (IQL)
* Decision Transformer

**Benefits for Coding:**

* Learn from existing code repositories
* No expensive online interaction
* Safety (no execution required)

Multi-Agent RL
--------------

Multiple agents learning simultaneously.

**Coordination:**

* Cooperative: Shared goals
* Competitive: Adversarial
* Mixed: Both cooperation and competition

**Applications:**

* Code review (reviewer vs. author)
* Collaborative development
* Testing (test generator vs. code generator)

Inverse RL
----------

Learn reward function from demonstrations.

**Use Cases:**

* Infer coding style preferences
* Learn quality criteria
* Discover best practices

Model-Based RL
--------------

Learn environment model for planning.

**Approaches:**

* World models
* Model-predictive control
* Dyna-Q

**Coding Applications:**

* Model code execution
* Predict test outcomes
* Plan refactoring sequences

Hierarchical RL
---------------

Multi-level decision making.

**Levels:**

* High-level: Task decomposition
* Mid-level: Sub-task execution
* Low-level: Code generation

**Benefits:**

* Handle complex tasks
* Transfer sub-policies
* Better exploration

Tools & Frameworks
==================

RL Libraries
------------

* **Stable Baselines3:** User-friendly RL implementations
* **Ray RLlib:** Scalable RL framework
* **TF-Agents:** TensorFlow RL library
* **CleanRL:** Simple implementations

LLM + RL
--------

* **TRL (Transformer RL):** Hugging Face library for LLM RL
* **OpenAI Gym:** Standard RL interface
* **Custom environments:** Code execution sandboxes

Debugging & Visualization
--------------------------

* TensorBoard
* Weights & Biases
* RLlib's visualization tools
* Custom logging

Common Pitfalls
===============

Reward Hacking
--------------

Agent exploits reward signal in unintended ways.

**Example:** Generate trivial code that passes weak tests

**Prevention:**

* Comprehensive reward design
* Robust test suites
* Human oversight
* Adversarial testing

Sparse Rewards
--------------

Rewards too infrequent for learning.

**Solutions:**

* Reward shaping
* Curriculum learning
* Intrinsic motivation
* Demonstration bootstrapping

High Variance
-------------

Training is unstable and inconsistent.

**Fixes:**

* Larger batch sizes
* Advantage normalization
* Multiple seeds
* Hyperparameter tuning

Slow Convergence
----------------

Training takes too long.

**Improvements:**

* Better initialization (SFT)
* Hyperparameter optimization
* More compute resources
* Algorithm selection

Best Practices
==============

1. **Start with imitation:** Begin with supervised learning
2. **Design rewards carefully:** Clear, aligned incentives
3. **Use PPO:** Robust, well-tested algorithm
4. **Monitor KL divergence:** Prevent policy collapse
5. **Evaluate thoroughly:** Multiple metrics and test sets
6. **Iterate rapidly:** Fast feedback loops
7. **Log everything:** Track all relevant metrics
8. **Checkpoint frequently:** Save progress regularly
9. **Start simple:** Basic setup before complexity
10. **Validate rewards:** Ensure they measure what you want

Further Reading
===============

Books
-----

* "Reinforcement Learning: An Introduction" (Sutton & Barto)
* "Deep Reinforcement Learning Hands-On" (Lapan)

Papers
------

* "Proximal Policy Optimization Algorithms" (Schulman et al.)
* "Training language models to follow instructions with human feedback" (Ouyang et al.)
* "Mastering the game of Go with deep neural networks" (Silver et al.)

Courses
-------

* David Silver's RL Course
* Berkeley Deep RL Course (CS 285)
* Spinning Up in Deep RL (OpenAI)

Resources
=========

Tutorials
---------

* OpenAI Spinning Up
* Stable Baselines3 documentation
* TRL examples

Code Examples
-------------

* RLHF implementations
* PPO from scratch
* Custom RL environments for code

See Also
========

* :doc:`post_training`
* :doc:`../evaluations`
* :doc:`../tools/frameworks`
