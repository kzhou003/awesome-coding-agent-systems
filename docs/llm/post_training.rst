====================
Post-Training
====================

Overview
========

Post-training techniques adapt pre-trained LLMs for specific coding tasks and improve their capabilities as coding agents. This section covers fine-tuning, alignment, and specialization methods.

Post-Training Paradigms
=======================

Supervised Fine-Tuning (SFT)
-----------------------------

Training on labeled examples to improve task performance.

**Process:**

1. Collect high-quality coding examples
2. Format as input-output pairs
3. Fine-tune model with supervised learning
4. Evaluate on held-out test set

**Data Sources:**

* GitHub repositories
* Competitive programming solutions
* Code review data
* Documentation and comments
* Synthetic data generation

**Typical Dataset Sizes:**

* Small-scale: 1K-10K examples
* Medium-scale: 10K-100K examples
* Large-scale: 100K-1M+ examples

**Example Training Format:**

.. code-block:: json

    {
      "instruction": "Write a function to find the longest common subsequence",
      "input": "def lcs(s1: str, s2: str) -> int:",
      "output": "    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    ..."
    }

Instruction Tuning
------------------

Fine-tuning on diverse instructions to improve instruction following.

**Key Aspects:**

* Diverse task coverage
* Natural language instructions
* Multi-turn conversations
* Varied output formats

**Popular Datasets:**

* Alpaca
* Dolly
* FLAN
* Self-Instruct generated data

Reinforcement Learning from Human Feedback (RLHF)
--------------------------------------------------

Aligning model behavior with human preferences through RL.

**Pipeline:**

1. **SFT:** Start with supervised fine-tuned model
2. **Reward Model:** Train on human preference comparisons
3. **RL Optimization:** Optimize policy using PPO/similar
4. **Evaluation:** Test aligned model

**Benefits:**

* Better alignment with user intent
* Improved code quality
* Reduced harmful outputs
* More helpful and honest responses

**Challenges:**

* Expensive human annotation
* Reward hacking
* Complexity of implementation
* Distribution shift

Direct Preference Optimization (DPO)
-------------------------------------

Simpler alternative to RLHF that directly optimizes preferences.

**Advantages over RLHF:**

* No separate reward model
* Simpler training pipeline
* More stable training
* Lower computational cost

**Process:**

1. Collect preference pairs (chosen vs. rejected)
2. Optimize model to prefer chosen responses
3. Uses binary cross-entropy style loss

**Formula:**

.. math::

    L_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]

Constitutional AI
-----------------

Training models to be helpful, harmless, and honest through self-critique.

**Process:**

1. Generate responses
2. Self-critique against principles
3. Revise based on critique
4. Train on revised responses

Code-Specific Training
=======================

Pre-training on Code
--------------------

Specialized pre-training or continued pre-training on code corpora.

**Data Sources:**

* GitHub (The Stack, StarCoder data)
* StackOverflow
* Documentation
* Technical books
* Jupyter notebooks

**Considerations:**

* License compliance
* Data deduplication
* Language distribution
* Quality filtering

Code Understanding Tasks
-------------------------

Tasks to improve code comprehension:

* **Code Summarization:** Generate descriptions
* **Code Search:** Match queries to code
* **Variable Naming:** Suggest meaningful names
* **Type Inference:** Predict types
* **Bug Detection:** Identify issues

Code Generation Tasks
---------------------

Tasks to improve code production:

* **Function Implementation:** Generate function bodies
* **Program Synthesis:** Create programs from specs
* **Code Completion:** Auto-complete code
* **Test Generation:** Create unit tests
* **Documentation:** Generate docstrings

Code Transformation Tasks
--------------------------

Tasks to improve code manipulation:

* **Refactoring:** Improve code structure
* **Translation:** Convert between languages
* **Repair:** Fix buggy code
* **Optimization:** Improve performance
* **Style Adaptation:** Match coding standards

Multi-Task Training
-------------------

Training on multiple tasks simultaneously.

**Benefits:**

* Better generalization
* Shared representations
* Transfer learning
* Robust capabilities

**Task Mixing Strategies:**

* Equal sampling
* Proportional to dataset size
* Temperature-based sampling
* Curriculum learning

Training Techniques
===================

Low-Rank Adaptation (LoRA)
---------------------------

Efficient fine-tuning by learning low-rank updates.

**Advantages:**

* Much fewer trainable parameters
* Lower memory requirements
* Faster training
* Multiple adapters for one base model

**Parameters:**

* Rank (r): Typically 8-64
* Alpha: Scaling factor
* Target modules: Which layers to adapt

**Example:**

.. code-block:: python

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, config)

QLoRA
~~~~~

Quantized LoRA for even more efficient training.

**Key Features:**

* 4-bit quantization of base model
* LoRA adapters in full precision
* Fits larger models in memory
* Minimal quality loss

Prefix Tuning
-------------

Learning task-specific prefix vectors.

**Concept:**

* Add learnable prefix tokens
* Keep base model frozen
* Prefix conditions generation

Adapter Layers
--------------

Small bottleneck layers inserted into model.

**Architecture:**

* Down-projection layer
* Non-linearity
* Up-projection layer
* Skip connection

Prompt Tuning
-------------

Learning soft prompts (continuous embeddings).

**Approach:**

* Prepend learnable embeddings
* Optimize embeddings, not model weights
* Very parameter efficient

Data Strategies
===============

Data Collection
---------------

**Sources:**

* Open-source repositories
* Competitive programming sites
* Stack Overflow
* Code review platforms
* Issue trackers

**Quality Signals:**

* Stars/forks
* Code review approval
* Test coverage
* Documentation quality
* Active maintenance

Data Cleaning
-------------

**Steps:**

1. License filtering
2. Deduplication
3. Quality scoring
4. Language detection
5. PII removal
6. Security vulnerability filtering

Data Augmentation
-----------------

**Techniques:**

* **Variable Renaming:** Systematically rename identifiers
* **Code Transformation:** Apply semantics-preserving edits
* **Comment Removal/Addition:** Vary comment presence
* **Docstring Generation:** Add documentation
* **Bug Injection:** Create buggy versions

Synthetic Data Generation
--------------------------

Using LLMs to generate training data.

**Approaches:**

* Self-instruct
* Backtranslation (code ↔ description)
* Evolutionary code generation
* Instruction diversification

**Quality Control:**

* Execution verification
* Static analysis
* Human review
* Automated filtering

Curriculum Learning
-------------------

Structured training progression.

**Strategies:**

* Start with simple problems
* Gradually increase complexity
* Task diversity progression
* Difficulty-based ordering

Training Infrastructure
=======================

Distributed Training
--------------------

**Approaches:**

* Data parallelism
* Model parallelism
* Pipeline parallelism
* ZeRO optimization

**Frameworks:**

* DeepSpeed
* Megatron-LM
* PyTorch FSDP
* JAX/FLAX

Hardware Considerations
-----------------------

**Options:**

* GPU clusters (A100, H100)
* TPU pods
* Cloud platforms (AWS, GCP, Azure)
* Custom infrastructure

**Trade-offs:**

* Cost vs. speed
* Memory vs. batch size
* Distributed overhead

Hyperparameter Tuning
----------------------

**Key Parameters:**

* Learning rate (typically 1e-5 to 1e-4 for fine-tuning)
* Batch size
* Number of epochs
* Warmup steps
* Weight decay
* Gradient clipping

**Search Strategies:**

* Grid search
* Random search
* Bayesian optimization
* Population-based training

Evaluation During Training
===========================

Validation Metrics
------------------

* Loss curves
* Pass@k on test set
* Code quality metrics
* Task-specific benchmarks

Early Stopping
--------------

Prevent overfitting:

* Monitor validation loss
* Patience parameter
* Best checkpoint saving

Checkpointing
-------------

Regular model saving:

* Save every N steps
* Keep top-k checkpoints
* Evaluation-based saving

Continuous Learning
-------------------

Ongoing model updates:

* Incremental training
* Online learning
* Periodic retraining

Alignment for Coding Agents
============================

Safety Alignment
----------------

**Goals:**

* Prevent generation of malicious code
* Avoid security vulnerabilities
* Refuse harmful requests
* Generate secure code by default

**Techniques:**

* Safety-focused SFT
* Constitutional AI
* Red-teaming
* Adversarial training

Helpfulness Alignment
---------------------

**Goals:**

* Follow instructions accurately
* Ask clarifying questions
* Provide explanations
* Handle ambiguity gracefully

**Training:**

* Human preference data
* Multi-turn conversation examples
* Instruction diversity

Tool Use Alignment
------------------

Training models to use tools effectively:

* Tool selection examples
* Successful tool use trajectories
* Error handling demonstrations
* Multi-tool coordination

Common Challenges
=================

Catastrophic Forgetting
-----------------------

Loss of general capabilities during specialization.

**Mitigation:**

* Mix general and specific data
* Regularization techniques
* Elastic weight consolidation
* Progressive neural networks

Overfitting
-----------

Model memorizes training data without generalizing.

**Prevention:**

* Sufficient data diversity
* Appropriate training duration
* Regularization
* Data augmentation

Distribution Shift
------------------

Training and deployment distributions differ.

**Solutions:**

* Diverse training data
* Domain adaptation
* Robust training objectives
* Continuous evaluation

Reward Hacking
--------------

In RLHF, model exploits reward model weaknesses.

**Mitigation:**

* Robust reward models
* KL divergence constraints
* Ensemble reward models
* Regular reward model updates

Tooling & Frameworks
====================

Training Frameworks
-------------------

* Hugging Face Transformers & TRL
* DeepSpeed
* Axolotl
* LLaMA Factory

RLHF Frameworks
---------------

* TRL (Transformer Reinforcement Learning)
* OpenAI's RLHF implementation
* Anthropic's approach
* Custom implementations

LoRA/PEFT Tools
---------------

* Hugging Face PEFT
* Microsoft LoRA
* Adapter Hub

Evaluation Suites
-----------------

* lm-evaluation-harness
* BigCode Evaluation Harness
* Custom evaluation scripts

Best Practices
==============

Data Quality
------------

1. Curate high-quality examples
2. Filter low-quality code
3. Ensure diversity
4. Verify correctness
5. Check licenses

Training Process
----------------

1. Start with strong base model
2. Use appropriate learning rates
3. Monitor for overfitting
4. Validate frequently
5. Keep best checkpoints

Evaluation
----------

1. Test on multiple benchmarks
2. Include real-world evaluation
3. Check for regressions
4. Measure multiple dimensions
5. Gather user feedback

Deployment
----------

1. Gradual rollout
2. A/B testing
3. Monitor metrics
4. Collect feedback
5. Iterate based on data

Research Directions
===================

* More efficient fine-tuning methods
* Better alignment techniques
* Automated data curation
* Multi-modal code training
* Continual learning for code
* Personalized fine-tuning

Resources
=========

Papers
------

* "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
* "Direct Preference Optimization"
* "LoRA: Low-Rank Adaptation of Large Language Models"
* "QLoRA: Efficient Finetuning of Quantized LLMs"

Datasets
--------

* The Stack
* CodeSearchNet
* APPS
* HumanEval
* MBPP

Tools
-----

* Hugging Face ecosystem
* DeepSpeed
* Weights & Biases (tracking)
* TensorBoard

See Also
========

* :doc:`benchmarking`
* :doc:`rl_basics`
* :doc:`../evaluations`
* :doc:`../performance/cost`
