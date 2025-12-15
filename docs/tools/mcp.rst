==================================
Model Context Protocol (MCP)
==================================

Overview
========

The Model Context Protocol (MCP) is an open protocol developed by Anthropic to standardize how applications provide context to Large Language Models (LLMs). It enables seamless integration between AI applications and data sources.

What is MCP?
============

Purpose
-------

MCP provides a standardized way for:

* **Context Provision:** Applications to expose data to LLMs
* **Tool Integration:** LLMs to use external tools and services
* **Resource Access:** Structured access to files, databases, APIs
* **Interoperability:** Standard interface across different systems

**Key Benefits:**

* Eliminates custom integrations for each data source
* Reduces vendor lock-in
* Promotes ecosystem growth
* Simplifies agent development

Architecture
------------

MCP follows a client-server architecture:

.. code-block:: text

    ┌─────────────────┐
    │   LLM Client    │  (Claude, ChatGPT, etc.)
    │   (MCP Client)  │
    └────────┬────────┘
             │ MCP Protocol
             │
    ┌────────▼────────┐
    │   MCP Server    │  (Filesystem, Database, API, etc.)
    │   (Data Source) │
    └─────────────────┘

**Components:**

* **Hosts:** Applications using LLMs (Claude Desktop, IDEs, custom apps)
* **Clients:** Protocol clients inside host applications
* **Servers:** Lightweight programs exposing resources/tools
* **Transport:** Communication layer (stdio, HTTP, WebSocket)

Core Concepts
=============

Resources
---------

Structured data sources that can be read by LLMs.

**Types:**

* Files and directories
* Database queries
* API endpoints
* Documentation
* Configuration files

**Resource Schema:**

.. code-block:: json

    {
      "uri": "file:///path/to/resource",
      "name": "Resource Name",
      "description": "Human-readable description",
      "mimeType": "application/json"
    }

**Example:**

.. code-block:: json

    {
      "uri": "database://main/users",
      "name": "Users Table",
      "description": "Application user records",
      "mimeType": "application/x-sql"
    }

Tools
-----

Functions that LLMs can invoke to perform actions.

**Tool Definition:**

.. code-block:: json

    {
      "name": "search_files",
      "description": "Search for files by pattern",
      "inputSchema": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "Search pattern (glob or regex)"
          },
          "path": {
            "type": "string",
            "description": "Directory to search in"
          }
        },
        "required": ["pattern"]
      }
    }

**Tool Invocation:**

.. code-block:: json

    {
      "name": "search_files",
      "arguments": {
        "pattern": "*.py",
        "path": "/src"
      }
    }

Prompts
-------

Reusable prompt templates with parameters.

**Prompt Schema:**

.. code-block:: json

    {
      "name": "code_review",
      "description": "Review code for quality and issues",
      "arguments": [
        {
          "name": "file_path",
          "description": "Path to file to review",
          "required": true
        },
        {
          "name": "focus_areas",
          "description": "Specific areas to focus on",
          "required": false
        }
      ]
    }

Sampling
--------

LLMs can request completions from the host application.

**Use Cases:**

* Agentic workflows
* Recursive task breakdown
* Sub-agent delegation

Protocol Specification
======================

Transport Layers
----------------

Standard I/O (stdio)
~~~~~~~~~~~~~~~~~~~~

Local process communication via stdin/stdout.

**Pros:**

* Simple to implement
* No network overhead
* Good for local tools

**Cons:**

* Process-bound
* No remote access

HTTP with Server-Sent Events (SSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Web-based protocol for remote servers.

**Pros:**

* Remote accessibility
* Firewall-friendly
* Standard web protocols

**Cons:**

* Network latency
* More complex setup

Custom Transports
~~~~~~~~~~~~~~~~~

Extensible to other transport mechanisms.

Message Format
--------------

JSON-RPC 2.0 based messaging.

**Request:**

.. code-block:: json

    {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {
        "name": "search_files",
        "arguments": {
          "pattern": "*.py"
        }
      }
    }

**Response:**

.. code-block:: json

    {
      "jsonrpc": "2.0",
      "id": 1,
      "result": {
        "content": [
          {
            "type": "text",
            "text": "Found 42 Python files..."
          }
        ]
      }
    }

**Error:**

.. code-block:: json

    {
      "jsonrpc": "2.0",
      "id": 1,
      "error": {
        "code": -32600,
        "message": "Invalid pattern syntax"
      }
    }

Protocol Methods
----------------

Server Methods
~~~~~~~~~~~~~~

* ``initialize``: Handshake and capability negotiation
* ``resources/list``: List available resources
* ``resources/read``: Read resource content
* ``tools/list``: List available tools
* ``tools/call``: Invoke a tool
* ``prompts/list``: List available prompts
* ``prompts/get``: Get prompt template

Client Methods
~~~~~~~~~~~~~~

* ``sampling/createMessage``: Request LLM completion
* ``roots/list``: List workspace roots

Notifications
~~~~~~~~~~~~~

* ``resources/updated``: Resource content changed
* ``tools/updated``: Tool definitions changed

Lifecycle
---------

1. **Connection:** Client connects to server
2. **Initialization:** Capability negotiation
3. **Discovery:** Client queries available resources/tools
4. **Operation:** Client invokes tools, reads resources
5. **Updates:** Server notifies of changes
6. **Shutdown:** Graceful connection close

Building MCP Servers
====================

Server Implementation
---------------------

**TypeScript/JavaScript:**

.. code-block:: typescript

    import { Server } from "@modelcontextprotocol/sdk/server/index.js";
    import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

    const server = new Server({
      name: "my-mcp-server",
      version: "1.0.0"
    });

    // Register tools
    server.setRequestHandler("tools/list", async () => ({
      tools: [
        {
          name: "search_files",
          description: "Search for files",
          inputSchema: { /* ... */ }
        }
      ]
    }));

    server.setRequestHandler("tools/call", async (request) => {
      const { name, arguments: args } = request.params;
      // Implement tool logic
      return { content: [/* ... */] };
    });

    const transport = new StdioServerTransport();
    await server.connect(transport);

**Python:**

.. code-block:: python

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    server = Server("my-mcp-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_files",
                description="Search for files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"}
                    }
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "search_files":
            # Implement search logic
            results = search_files(arguments["pattern"])
            return [TextContent(type="text", text=results)]

    async def main():
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1])

Common Server Patterns
----------------------

Filesystem Server
~~~~~~~~~~~~~~~~~

Expose file system access to LLMs.

**Capabilities:**

* List directories
* Read files
* Search content
* Watch for changes

Database Server
~~~~~~~~~~~~~~~

Provide structured data access.

**Features:**

* Query execution
* Schema inspection
* Result formatting
* Connection pooling

API Server
~~~~~~~~~~

Wrap external APIs as MCP tools.

**Responsibilities:**

* Authentication
* Rate limiting
* Response transformation
* Error handling

Git Server
~~~~~~~~~~

Integrate version control.

**Operations:**

* Commit history
* Diff viewing
* Branch information
* Blame annotations

Testing Servers
---------------

**Tools:**

* MCP Inspector (official debugging tool)
* Unit tests with mock clients
* Integration tests with real LLMs

**Testing Checklist:**

* Tool invocation
* Error handling
* Resource reading
* Update notifications
* Concurrent requests

Using MCP Clients
=================

Official Clients
----------------

Claude Desktop
~~~~~~~~~~~~~~

Native MCP support in Claude Desktop app.

**Configuration:**

.. code-block:: json

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
        },
        "database": {
          "command": "python",
          "args": ["mcp_database_server.py"],
          "env": {
            "DATABASE_URL": "postgresql://..."
          }
        }
      }
    }

**Location:**

* macOS: ``~/Library/Application Support/Claude/claude_desktop_config.json``
* Windows: ``%APPDATA%\Claude\claude_desktop_config.json``

Custom Applications
-------------------

**Integrating MCP:**

.. code-block:: typescript

    import { Client } from "@modelcontextprotocol/sdk/client/index.js";
    import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

    const transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    });

    const client = new Client({
      name: "my-client",
      version: "1.0.0"
    });

    await client.connect(transport);

    // List tools
    const tools = await client.listTools();

    // Call tool
    const result = await client.callTool({
      name: "search_files",
      arguments: { pattern: "*.py" }
    });

MCP Ecosystem
=============

Official Servers
----------------

* **@modelcontextprotocol/server-filesystem:** File system access
* **@modelcontextprotocol/server-github:** GitHub integration
* **@modelcontextprotocol/server-git:** Git repository access
* **@modelcontextprotocol/server-postgres:** PostgreSQL database
* **@modelcontextprotocol/server-sqlite:** SQLite database
* **@modelcontextprotocol/server-puppeteer:** Browser automation

Community Servers
-----------------

* AWS services integration
* Google Cloud Platform tools
* Slack and communication platforms
* Monitoring and observability tools
* Development tools (Docker, Kubernetes)

Client Implementations
----------------------

* **Claude Desktop:** Official Anthropic client
* **Continue:** VS Code extension
* **Zed:** Text editor integration
* Custom applications

Use Cases for Coding Agents
============================

Code Search & Analysis
----------------------

**MCP Tools:**

* Semantic code search
* AST parsing
* Dependency analysis
* Symbol lookup

Development Environment
-----------------------

**Integration:**

* IDE features via MCP
* Build system access
* Test runner integration
* Debugger interface

Documentation Access
--------------------

**Resources:**

* API documentation
* Internal wikis
* Code comments
* Architecture diagrams

Version Control
---------------

**Operations:**

* Git commands via MCP tools
* Commit history access
* Diff analysis
* Branch management

External Services
-----------------

**Examples:**

* CI/CD platforms
* Issue trackers
* Code review systems
* Deployment tools

Security Considerations
=======================

Access Control
--------------

* Validate tool inputs
* Limit file system access
* Authenticate API requests
* Implement permission systems

Sandboxing
----------

* Isolate server processes
* Restrict system calls
* Limit resource usage
* Monitor server behavior

Data Privacy
------------

* Sanitize sensitive data
* Audit logging
* Secure communication
* Comply with regulations

Rate Limiting
-------------

* Prevent abuse
* Manage costs
* Ensure availability
* Fair resource allocation

Best Practices
==============

Server Development
------------------

1. **Clear Documentation:** Describe tools and resources thoroughly
2. **Error Handling:** Provide meaningful error messages
3. **Validation:** Validate all inputs
4. **Testing:** Comprehensive test coverage
5. **Logging:** Detailed operational logs
6. **Versioning:** Semantic versioning for servers

Tool Design
-----------

1. **Single Responsibility:** Each tool does one thing well
2. **Descriptive Names:** Clear, self-explanatory tool names
3. **Rich Schemas:** Detailed input/output specifications
4. **Examples:** Provide usage examples
5. **Idempotency:** Make tools safe to retry

Client Integration
------------------

1. **Error Recovery:** Handle server failures gracefully
2. **Caching:** Cache tool and resource lists
3. **Timeouts:** Implement reasonable timeouts
4. **Monitoring:** Track usage and performance
5. **Updates:** Handle server capability changes

Future of MCP
=============

Emerging Patterns
-----------------

* Federated MCP servers
* MCP-to-MCP communication
* Standardized tool catalogs
* Cross-platform tool sharing

Research Directions
-------------------

* Automatic server generation
* Semantic tool discovery
* Adaptive context selection
* Performance optimization

Resources
=========

Official Documentation
----------------------

* MCP Specification: https://spec.modelcontextprotocol.io
* GitHub Repository: https://github.com/modelcontextprotocol
* TypeScript SDK: @modelcontextprotocol/sdk
* Python SDK: mcp

Community
---------

* GitHub Discussions
* Discord server
* Example implementations
* Server registry

Tutorials
---------

* Building your first MCP server
* Integrating MCP in applications
* Advanced MCP patterns
* Security best practices

See Also
========

* :doc:`a2a`
* :doc:`frameworks`
* :doc:`patterns`
* :doc:`../llm/tool_selection`
