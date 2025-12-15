====================
Agent Deployments
====================

Overview
========

Deploying coding agents to production requires careful consideration of infrastructure, scalability, reliability, and user experience. This section covers deployment architectures, strategies, and best practices.

Deployment Considerations
==========================

Requirements Analysis
---------------------

**Key Questions:**

* What is the expected load?
* What are latency requirements?
* What is the budget?
* What are security requirements?
* What is the target environment (cloud, on-prem, edge)?

**Trade-offs:**

* Performance vs. cost
* Latency vs. throughput
* Simplicity vs. flexibility
* Self-hosted vs. managed services

Deployment Environments
-----------------------

Cloud
~~~~~

**Providers:**

* AWS
* Google Cloud Platform
* Microsoft Azure
* Oracle Cloud

**Pros:**

* Scalability
* Managed services
* Global reach
* Pay-as-you-go

**Cons:**

* Ongoing costs
* Vendor lock-in
* Data sovereignty concerns

On-Premises
~~~~~~~~~~~

**Pros:**

* Full control
* Data privacy
* No ongoing cloud costs
* Compliance easier

**Cons:**

* Hardware investment
* Maintenance burden
* Limited scalability
* Expertise required

Hybrid
~~~~~~

Combination of cloud and on-premises.

**Use Cases:**

* Sensitive data on-prem
* Scaling burst to cloud
* Multi-region deployment

Edge/Local
~~~~~~~~~~

Agent runs on user's device.

**Pros:**

* No latency
* Privacy
* Offline capability
* No per-query costs

**Cons:**

* Limited compute
* Deployment complexity
* Version management

Deployment Architectures
=========================

Simple API Service
------------------

Single service exposing agent via API.

**Architecture:**

.. code-block:: text

    Client → Load Balancer → Agent API Server → LLM Provider

**Implementation:**

.. code-block:: python

    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class AgentRequest(BaseModel):
        task: str
        context: dict

    class AgentResponse(BaseModel):
        result: str
        status: str

    @app.post("/agent/execute")
    async def execute_agent(request: AgentRequest) -> AgentResponse:
        agent = initialize_agent()

        try:
            result = await agent.execute(
                task=request.task,
                context=request.context
            )

            return AgentResponse(
                result=result,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                result=str(e),
                status="error"
            )

**Pros:**

* Simple to implement
* Easy to understand
* Quick to deploy

**Cons:**

* Limited scalability
* No fault tolerance
* Stateless (needs external state management)

Microservices Architecture
--------------------------

Agent functionality split across services.

**Architecture:**

.. code-block:: text

    Client
      ↓
    API Gateway
      ↓
    ┌─────────┬──────────┬────────────┐
    │ Planner │ Executor │  Evaluator │
    └─────────┴──────────┴────────────┘
         ↓          ↓            ↓
    ┌─────────────────────────────────┐
    │      Message Queue (Kafka)      │
    └─────────────────────────────────┘
         ↓
    ┌─────────┬──────────┬────────────┐
    │ Storage │  Vector  │    LLM     │
    │ Service │    DB    │  Service   │
    └─────────┴──────────┴────────────┘

**Components:**

* **API Gateway:** Request routing, authentication
* **Planning Service:** Task decomposition
* **Execution Service:** Action execution
* **Evaluation Service:** Result validation
* **Storage Service:** Persistent data
* **Vector DB:** Semantic search
* **LLM Service:** Model inference

**Pros:**

* Scalable
* Independent scaling
* Technology flexibility
* Fault isolation

**Cons:**

* Complex
* Network overhead
* Distributed system challenges

Serverless Architecture
-----------------------

Event-driven, auto-scaling functions.

**Architecture:**

.. code-block:: text

    Client → API Gateway → Lambda/Cloud Functions
                              ↓
                         Event Bus
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Worker Functions    Storage & Databases

**Implementation (AWS Lambda):**

.. code-block:: python

    import json

    def lambda_handler(event, context):
        # Parse request
        body = json.loads(event['body'])
        task = body['task']

        # Initialize agent
        agent = initialize_agent()

        # Execute
        result = agent.execute(task)

        # Return response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'result': result
            })
        }

**Pros:**

* Auto-scaling
* Pay per use
* No server management
* High availability

**Cons:**

* Cold start latency
* Limited execution time
* Vendor lock-in
* Debugging challenges

Container-Based Deployment
--------------------------

Using Docker and Kubernetes.

**Docker Setup:**

.. code-block:: dockerfile

    FROM python:3.11-slim

    WORKDIR /app

    # Install dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy code
    COPY . .

    # Expose port
    EXPOSE 8000

    # Run application
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

**Kubernetes Deployment:**

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: coding-agent
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: coding-agent
      template:
        metadata:
          labels:
            app: coding-agent
        spec:
          containers:
          - name: agent
            image: myregistry/coding-agent:latest
            ports:
            - containerPort: 8000
            resources:
              requests:
                memory: "2Gi"
                cpu: "1000m"
              limits:
                memory: "4Gi"
                cpu: "2000m"
            env:
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: api-key
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: coding-agent-service
    spec:
      selector:
        app: coding-agent
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
      type: LoadBalancer

**Pros:**

* Portable
* Consistent environments
* Orchestration (Kubernetes)
* Easy scaling

**Cons:**

* Learning curve
* Resource overhead
* Complexity

Infrastructure Components
=========================

Load Balancing
--------------

Distribute traffic across instances.

**Options:**

* **Cloud Load Balancers:** AWS ELB, GCP Load Balancing
* **Software:** Nginx, HAProxy
* **Service Mesh:** Istio, Linkerd

**Configuration (Nginx):**

.. code-block:: nginx

    upstream agent_backend {
        least_conn;
        server agent1:8000;
        server agent2:8000;
        server agent3:8000;
    }

    server {
        listen 80;

        location /api/agent {
            proxy_pass http://agent_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }

Caching
-------

Reduce latency and costs for repeated queries.

**Caching Strategies:**

Response Caching
~~~~~~~~~~~~~~~~

Cache complete agent responses.

.. code-block:: python

    from functools import lru_cache
    import redis

    redis_client = redis.Redis(host='localhost', port=6379)

    async def get_cached_response(query: str):
        # Check cache
        cached = redis_client.get(f"agent:response:{hash(query)}")
        if cached:
            return json.loads(cached)

        # Generate response
        response = await agent.execute(query)

        # Store in cache
        redis_client.setex(
            f"agent:response:{hash(query)}",
            3600,  # 1 hour TTL
            json.dumps(response)
        )

        return response

Embedding Caching
~~~~~~~~~~~~~~~~~

Cache embeddings for repeated queries.

Context Caching
~~~~~~~~~~~~~~~

Cache common context/prompts.

Message Queues
--------------

Async processing and decoupling.

**Technologies:**

* RabbitMQ
* Apache Kafka
* AWS SQS
* Redis Streams

**Example (RabbitMQ):**

.. code-block:: python

    import pika

    # Producer
    def submit_task(task: dict):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        channel = connection.channel()

        channel.queue_declare(queue='agent_tasks', durable=True)

        channel.basic_publish(
            exchange='',
            routing_key='agent_tasks',
            body=json.dumps(task),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        connection.close()

    # Consumer
    def process_tasks():
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        channel = connection.channel()

        channel.queue_declare(queue='agent_tasks', durable=True)

        def callback(ch, method, properties, body):
            task = json.loads(body)
            agent = initialize_agent()
            result = agent.execute(task)
            # Store result
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(
            queue='agent_tasks',
            on_message_callback=callback
        )

        channel.start_consuming()

State Management
----------------

Maintain agent state across requests.

**Options:**

* Redis (in-memory)
* PostgreSQL (relational)
* MongoDB (document)
* DynamoDB (NoSQL)

**Session State:**

.. code-block:: python

    class SessionManager:
        def __init__(self, redis_client):
            self.redis = redis_client

        def save_session(self, session_id: str, state: dict):
            self.redis.setex(
                f"session:{session_id}",
                3600,  # 1 hour
                json.dumps(state)
            )

        def load_session(self, session_id: str) -> dict:
            data = self.redis.get(f"session:{session_id}")
            return json.loads(data) if data else {}

Monitoring & Observability
===========================

Logging
-------

Structured logging for debugging and analysis.

**Implementation:**

.. code-block:: python

    import logging
    import json
    from datetime import datetime

    class StructuredLogger:
        def __init__(self, name: str):
            self.logger = logging.getLogger(name)

        def log_request(
            self,
            request_id: str,
            task: str,
            user_id: str
        ):
            self.logger.info(json.dumps({
                "event": "request",
                "request_id": request_id,
                "task": task,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }))

        def log_execution(
            self,
            request_id: str,
            step: str,
            duration_ms: float
        ):
            self.logger.info(json.dumps({
                "event": "execution",
                "request_id": request_id,
                "step": step,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            }))

**Tools:**

* ELK Stack (Elasticsearch, Logstash, Kibana)
* Splunk
* Datadog
* CloudWatch Logs

Metrics
-------

Track performance and health metrics.

**Key Metrics:**

* Request rate
* Latency (p50, p95, p99)
* Error rate
* Token usage
* Cost per request
* Success rate

**Implementation (Prometheus):**

.. code-block:: python

    from prometheus_client import Counter, Histogram, start_http_server

    # Define metrics
    request_count = Counter(
        'agent_requests_total',
        'Total agent requests',
        ['status']
    )

    request_duration = Histogram(
        'agent_request_duration_seconds',
        'Agent request duration'
    )

    token_usage = Counter(
        'agent_tokens_total',
        'Total tokens used',
        ['type']
    )

    # Use metrics
    @request_duration.time()
    async def execute_agent(task: str):
        try:
            result = await agent.execute(task)
            request_count.labels(status='success').inc()
            token_usage.labels(type='input').inc(result.input_tokens)
            token_usage.labels(type='output').inc(result.output_tokens)
            return result
        except Exception as e:
            request_count.labels(status='error').inc()
            raise

Tracing
-------

Track request flow through distributed system.

**Implementation (OpenTelemetry):**

.. code-block:: python

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter

    # Setup tracer
    tracer_provider = TracerProvider()
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)

    # Use tracing
    async def execute_agent(task: str):
        with tracer.start_as_current_span("execute_agent") as span:
            span.set_attribute("task.type", task.type)

            with tracer.start_as_current_span("planning"):
                plan = await planner.create_plan(task)

            with tracer.start_as_current_span("execution"):
                result = await executor.execute(plan)

            return result

**Tools:**

* Jaeger
* Zipkin
* AWS X-Ray
* Google Cloud Trace

Alerting
--------

Notify on issues and anomalies.

**Alert Rules:**

.. code-block:: yaml

    groups:
      - name: agent_alerts
        rules:
          - alert: HighErrorRate
            expr: rate(agent_requests_total{status="error"}[5m]) > 0.05
            for: 5m
            annotations:
              summary: "High error rate detected"

          - alert: HighLatency
            expr: histogram_quantile(0.95, agent_request_duration_seconds) > 5
            for: 5m
            annotations:
              summary: "95th percentile latency > 5s"

          - alert: HighCost
            expr: rate(agent_tokens_total[1h]) > 1000000
            for: 1h
            annotations:
              summary: "Token usage exceeds threshold"

Security
========

Authentication & Authorization
------------------------------

Verify and authorize users.

**API Key Authentication:**

.. code-block:: python

    from fastapi import Security, HTTPException
    from fastapi.security import APIKeyHeader

    api_key_header = APIKeyHeader(name="X-API-Key")

    async def verify_api_key(api_key: str = Security(api_key_header)):
        if not is_valid_api_key(api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return api_key

    @app.post("/agent/execute")
    async def execute_agent(
        request: AgentRequest,
        api_key: str = Security(verify_api_key)
    ):
        # Execute agent
        pass

**OAuth 2.0:**

.. code-block:: python

    from fastapi import Depends
    from fastapi.security import OAuth2PasswordBearer

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    async def get_current_user(token: str = Depends(oauth2_scheme)):
        user = verify_token(token)
        if not user:
            raise HTTPException(status_code=401)
        return user

Rate Limiting
-------------

Prevent abuse and manage costs.

**Implementation:**

.. code-block:: python

    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.post("/agent/execute")
    @limiter.limit("10/minute")
    async def execute_agent(request: Request, agent_request: AgentRequest):
        # Execute agent
        pass

Input Validation
----------------

Sanitize and validate user inputs.

.. code-block:: python

    from pydantic import BaseModel, validator, Field

    class AgentRequest(BaseModel):
        task: str = Field(..., max_length=10000)
        context: dict

        @validator('task')
        def validate_task(cls, v):
            # Check for malicious patterns
            if contains_malicious_patterns(v):
                raise ValueError("Invalid task content")
            return v

        @validator('context')
        def validate_context(cls, v):
            # Limit context size
            if len(json.dumps(v)) > 100000:
                raise ValueError("Context too large")
            return v

Secrets Management
------------------

Secure handling of API keys and credentials.

**Options:**

* AWS Secrets Manager
* HashiCorp Vault
* Google Secret Manager
* Azure Key Vault

**Example (AWS Secrets Manager):**

.. code-block:: python

    import boto3

    def get_secret(secret_name: str) -> str:
        client = boto3.client('secretsmanager')
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']

    # Usage
    llm_api_key = get_secret("production/llm_api_key")

Deployment Strategies
=====================

Blue-Green Deployment
---------------------

Maintain two environments, switch traffic.

**Process:**

1. Blue environment serves production
2. Deploy new version to Green environment
3. Test Green environment
4. Switch traffic to Green
5. Blue becomes standby

**Benefits:**

* Zero downtime
* Easy rollback
* Reduced risk

Canary Deployment
-----------------

Gradually roll out to subset of users.

**Process:**

1. Deploy new version to small % of servers
2. Monitor metrics
3. If successful, increase %
4. Eventually 100%

**Implementation:**

.. code-block:: python

    def route_request(request: Request):
        user_id = get_user_id(request)
        canary_percentage = 10  # 10% canary

        if hash(user_id) % 100 < canary_percentage:
            return canary_agent.execute(request)
        else:
            return stable_agent.execute(request)

Rolling Deployment
------------------

Gradual instance-by-instance update.

**Kubernetes:**

.. code-block:: yaml

    spec:
      replicas: 10
      strategy:
        type: RollingUpdate
        rollingUpdate:
          maxSurge: 2
          maxUnavailable: 1

A/B Testing
-----------

Compare versions with user subsets.

See :doc:`evaluations` for implementation details.

Scaling Strategies
==================

Vertical Scaling
----------------

Increase resources of single instance.

**Pros:**

* Simple
* No code changes

**Cons:**

* Limited
* Expensive
* Single point of failure

Horizontal Scaling
------------------

Add more instances.

**Auto-scaling (Kubernetes HPA):**

.. code-block:: yaml

    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: coding-agent-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: coding-agent
      minReplicas: 2
      maxReplicas: 20
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Resource
        resource:
          name: memory
          target:
            type: Utilization
            averageUtilization: 80

Model Optimization
------------------

Improve inference performance.

**Techniques:**

* Quantization (4-bit, 8-bit)
* Batching
* Caching
* Prompt optimization
* Smaller models

Best Practices
==============

1. **Start Simple:** Begin with simple architecture
2. **Monitor Everything:** Comprehensive observability
3. **Automate:** CI/CD pipelines
4. **Test Thoroughly:** Staging environments
5. **Plan for Failure:** Graceful degradation
6. **Version Everything:** Code, models, configs
7. **Document:** Architecture and runbooks
8. **Security First:** Defense in depth
9. **Cost Awareness:** Monitor and optimize
10. **User Experience:** Prioritize latency and reliability

Resources
=========

Cloud Platforms
---------------

* AWS: https://aws.amazon.com
* GCP: https://cloud.google.com
* Azure: https://azure.microsoft.com

Tools
-----

* Docker: https://www.docker.com
* Kubernetes: https://kubernetes.io
* Terraform: https://www.terraform.io
* Prometheus: https://prometheus.io
* Grafana: https://grafana.com

See Also
========

* :doc:`performance/latency`
* :doc:`performance/cost`
* :doc:`security`
* :doc:`evaluations`
