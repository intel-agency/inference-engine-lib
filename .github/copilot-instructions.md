---
description: Entry point for GitHub Copilot custom instructions
scope: global
role: System Orchestrator
---

<instructions>
  <overview>
    This file serves as the bootstrap entry point for the AI agent's instruction set.
    It defines the location of core modules, the protocol for loading remote instructions, and the single source of truth policy.
  </overview>

  <configuration>
    <!-- BRANCH PARAMETER: Change this value to load instructions from a different branch -->
    <!-- Valid values: main, optimization, feature/*, or any valid branch name -->
    <branch>main</branch>
  </configuration>

  <instruction_source>
    <repository>
      <name>nam20485/agent-instructions</name>
      <url>https://github.com/nam20485/agent-instructions/tree/{branch}</url>
      <branch>{branch}</branch>
    </repository>
    <guidance>
      Start with the Core Instructions linked below. Follow links to other modules as required by the user's request.
      All remote URLs use the branch specified in the configuration section above.
    </guidance>
  </instruction_source>

  <module_registry>
    <module type="core" required="true">
      <name>Core Instructions</name>
      <link>https://github.com/nam20485/agent-instructions/blob/{branch}/ai_instruction_modules/ai-core-instructions.md</link>
      <description>The foundational behaviors and rules for the agent.</description>
    </module>

    <module type="local" required="true">
      <name>Local AI Instructions</name>
      <path>../local_ai_instruction_modules</path>
      <description>Context-specific instructions located in the local workspace.</description>
    </module>

    <module type="dynamic workflow" required="true">
      <name>Dynamic Workflow Orchestration</name>
      <path>../local_ai_instruction_modules/ai-dynamic-workflows.md</path>
      <description>Protocol for resolving workflows from the remote canonical repository.</description>
    </module>

    <module type="workflow assignment" required="true">
      <name>Workflow Assignments</name>
      <path>../local_ai_instruction_modules/ai-workflow-assignments.md</path>
      <description>Index of active workflow assignments by shortId.</description>
    </module>

    <module type="optional">
      <name>Terminal Commands</name>
      <path>../local_ai_instruction_modules/ai-terminal-commands.md</path>
      <description>Reference for terminal operations and GitHub CLI usage.</description>
    </module>
  </module_registry>

  <loading_protocol>
    <rule id="branch_resolution">
      <description>Resolving the active branch</description>
      <instruction>
        Read the branch value from the configuration section at the top of this file.
        Replace all `{branch}` placeholders in URLs with this value.
        Default: `main` if not specified.
      </instruction>
    </rule>

    <rule id="remote_access">
      <description>Accessing files in the remote repository</description>
      <instruction>
        Always use the RAW URL to read file contents. Do not use the GitHub UI URL.
      </instruction>
    </rule>

    <algorithm name="url_translation">
      <step>Read the configured branch from `<configuration><branch>`.</step>
      <step>Identify the GitHub UI URL (e.g., `https://github.com/.../blob/{branch}/...`).</step>
      <step>Replace `https://github.com/` with `https://raw.githubusercontent.com/`.</step>
      <step>Remove `blob/` from the path.</step>
      <step>Substitute `{branch}` with the configured branch value.</step>
      <step>Result: `https://raw.githubusercontent.com/.../{branch}/...`</step>
    </algorithm>

    <examples>
      <example title="Default (main branch)">
        <config_branch>main</config_branch>
        <input>https://github.com/nam20485/agent-instructions/blob/{branch}/ai_instruction_modules/ai-core-instructions.md</input>
        <output>https://raw.githubusercontent.com/nam20485/agent-instructions/main/ai_instruction_modules/ai-core-instructions.md</output>
      </example>
      <example title="Optimization branch">
        <config_branch>optimization</config_branch>
        <input>https://github.com/nam20485/agent-instructions/blob/{branch}/ai_instruction_modules/ai-core-instructions.md</input>
        <output>https://raw.githubusercontent.com/nam20485/agent-instructions/optimization/ai_instruction_modules/ai-core-instructions.md</output>
      </example>
    </examples>
  </loading_protocol>

  <policy name="single_source_of_truth">
    <statement>
      The remote canonical repository is the ONLY authoritative source for dynamic workflows and workflow assignments.
    </statement>
    <rules>
      <rule>Do not use local mirrors or cached plans to derive steps.</rule>
      <rule>Fetch and execute directly from the remote canonical URLs.</rule>
      <rule>Changes in the remote repo take effect immediately.</rule>
    </rules>
  </policy>
  <tool_use_instructions>
    <instruction id="querying_microsoft_documentation">
      <applyTo>**</applyTo>
      <title>Querying Microsoft Documentation</title>
      <tools>
        <tool>microsoft_docs_search</tool>
        <tool>microsoft_docs_fetch</tool>
        <tool>microsoft_code_sample_search</tool>
      </tools>
      <guidance>
        These MCP tools can search and fetch Microsoft's latest official documentation and code samples, which may be newer or more detailed than model training data.
      </guidance>
      <guidance>
        For specific and narrowly defined questions involving native Microsoft technologies (C#, F#, ASP.NET Core, Microsoft.Extensions, NuGet, Entity Framework, and the dotnet runtime), use these tools for research.
        When writing code, prioritize using information retrieved from these tools to ensure accuracy and up-to-date practices, especially for newer features or libraries.
        Before writing any code involving Microsoft technologies, always check for relevant documentation or code samples using these tools to ensure the most current and accurate information is being used.
      </guidance>
    </instruction>
    <instruction id="sequential_thinking_default_usage">
      <applyTo>*</applyTo>
      <title>Sequential Thinking for Complex Problem Solving</title>
      <tools>
        <tool>sequential_thinking</tool>
      </tools>
      <guidance>
        Use sequential thinking for all requests except the most trivial, single-step requests (for example, minimal formatting changes or direct one-line lookups).
      </guidance>
      <guidance>
        The sequential_thinking tool enables structured, step-by-step problem analysis with the ability to revise, branch, and adjust reasoning paths dynamically.
        Use this tool when:
        - Breaking down complex problems into manageable steps
        - Planning and design work that may require revision
        - Analyzing situations where the full scope is unclear initially
        - Conducting analysis that might need course correction
        - Making architectural or design decisions
        - Encountering unexpected issues or errors that require systematic debugging
        - Filtering out irrelevant information while maintaining context
        - Working through tasks that need to maintain context over multiple steps
      </guidance>
      <guidance>
        The tool supports dynamic thinking processes by allowing you to:
        - Revise previous thoughts when new understanding emerges
        - Branch into alternative reasoning paths
        - Adjust the estimated total number of thoughts as complexity becomes apparent
        - Mark when additional thinking steps are needed beyond original estimates
      </guidance>
    </instruction>
    <instruction id="memory_default_usage">
      <applyTo>*</applyTo>
      <title>Knowledge Graph Memory for Context Persistence</title>
      <tools>
        <tool>create_entities</tool>
        <tool>create_relations</tool>
        <tool>add_observations</tool>
        <tool>delete_entities</tool>
        <tool>delete_observations</tool>
        <tool>delete_relations</tool>
        <tool>read_graph</tool>
        <tool>search_nodes</tool>
        <tool>open_nodes</tool>
      </tools>
      <guidance>
        Use memory for all requests except the most trivial, single-step requests, and persist relevant user/project context when it helps future task continuity.
      </guidance>
      <guidance>
        The memory MCP server provides a persistent knowledge graph for storing and retrieving context across conversations.
        Store information about:
        - User preferences, communication styles, and working patterns
        - Project-specific configurations, patterns, and conventions
        - Technical decisions, rationale, and architectural choices
        - Recurring challenges, solutions, and workarounds discovered
        - Team members, their roles, and areas of expertise
        - Tool configurations, CLI paths, authentication mechanisms
        - Build patterns, deployment strategies, and environment setups
      </guidance>
      <guidance>
        Knowledge graph concepts:
        - **Entities**: Primary nodes with a name, type (e.g., "person", "project", "tool"), and observations
        - **Relations**: Directed connections between entities in active voice (e.g., "works_at", "depends_on", "configures")
        - **Observations**: Atomic facts stored as strings attached to entities (one fact per observation)
      </guidance>
      <guidance>
        Tool usage patterns:
        - Use create_entities to establish new concepts, people, projects, or tools
        - Use create_relations to connect related entities and build context
        - Use add_observations to record facts, preferences, or discoveries about entities
        - Use search_nodes to find relevant context based on keywords across names, types, and observations
        - Use open_nodes to retrieve specific entities by name for detailed information
        - Use read_graph to get a complete view of all stored knowledge when planning or reviewing
        - Use delete operations to clean up outdated or incorrect information
      </guidance>
      <guidance>
        At the start of complex tasks, search or read relevant memory to leverage previous learnings.
        After completing significant work, update memory with new patterns, configurations, or insights discovered.
      </guidance>
    </instruction>
  </tool_use_instructions>
</instructions>
