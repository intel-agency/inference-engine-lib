---
description: Project instructions for coding agents
scope: repository
role: InferenceEngine Library Agent
---

<instructions>
  <purpose>
    <summary>
      This project is a cross-platform .NET 10 inference library that wraps ONNX Runtime behind a generic
      `IInferenceEngine&lt;TInput, TOutput&gt;` interface with automatic hardware acceleration:
      DirectML on Windows, CoreML on macOS, and ROCm on Linux x64 (with CPU fallback).
    </summary>
    <guidance>
      Use this file as the default operating guide for coding agents working in this repository.
    </guidance>
  </purpose>

  <tech_stack>
    <item>C# 13 / .NET 10</item>
    <item>ONNX Runtime</item>
    <item>Microsoft.Extensions.Logging.Abstractions</item>
    <item>xUnit + Xunit.SkippableFact + Coverlet</item>
    <item>Trunk CLI for lint/check workflows</item>
  </tech_stack>

  <repository_map>
    <entry>
      <path>InferenceEngine.Core/</path>
      <description>Public API, base classes, options, model providers, and model registry plumbing.</description>
    </entry>
    <entry>
      <path>InferenceEngine.Examples/</path>
      <description>Reference engines such as `FaceDetectionEngine`.</description>
    </entry>
    <entry>
      <path>InferenceEngine.Tests/</path>
      <description>Unit/integration tests for Core and Examples.</description>
    </entry>
    <entry>
      <path>.trunk/trunk.yaml</path>
      <description>Trunk linter configuration (including `dotnet-format`).</description>
    </entry>
    <entry>
      <path>scripts/agent-self-check.sh</path>
      <description>One-command lint/build/test check for AI-authored changes.</description>
    </entry>
    <entry>
      <path>.github/workflows/build-test.yml</path>
      <description>Cross-platform CI build/test workflow.</description>
    </entry>
    <entry>
      <path>.github/workflows/publish.yml</path>
      <description>Release branch packaging and NuGet publication workflow.</description>
    </entry>
  </repository_map>

  <key_types>
    <entry>
      <name>IInferenceEngine&lt;TInput, TOutput&gt;</name>
      <path>InferenceEngine.Core/IInferenceEngine.cs</path>
      <description>Generic contract with `LoadAsync` and `PredictAsync`.</description>
    </entry>
    <entry>
      <name>BaseInferenceEngine&lt;TInput, TOutput&gt;</name>
      <path>InferenceEngine.Core/BaseInferenceEngine.cs</path>
      <description>ONNX session lifecycle, hardware acceleration, warmup, and disposal.</description>
    </entry>
    <entry>
      <name>InferenceEngineOptions</name>
      <path>InferenceEngine.Core/InferenceEngineOptions.cs</path>
      <description>Configuration record for GPU toggle, device ID, warmup, and model provider.</description>
    </entry>
    <entry>
      <name>IModelProvider</name>
      <path>InferenceEngine.Core/ModelProviders/IModelProvider.cs</path>
      <description>Abstraction for model acquisition.</description>
    </entry>
    <entry>
      <name>LocalModelProvider</name>
      <path>InferenceEngine.Core/ModelProviders/LocalModelProvider.cs</path>
      <description>Supplies model from local filesystem path.</description>
    </entry>
    <entry>
      <name>DownloadingModelProvider</name>
      <path>InferenceEngine.Core/ModelProviders/DownloadingModelProvider.cs</path>
      <description>Downloads model files with cache and fallback URLs.</description>
    </entry>
    <entry>
      <name>ModelRegistry</name>
      <path>InferenceEngine.Core/ModelProviders/ModelRegistry.cs</path>
      <description>Maps model names to download URLs from `models.json`.</description>
    </entry>
    <entry>
      <name>FaceDetectionEngine</name>
      <path>InferenceEngine.Examples/FaceDetectionEngine.cs</path>
      <description>YOLO11 object detection example with SkiaSharp preprocessing.</description>
    </entry>
  </key_types>

  <coding_conventions>
    <language_framework>
      <rule>Use C# 13 / .NET 10 features that match existing repository style.</rule>
      <rule>Nullable reference types are enabled and expected in all new code.</rule>
      <rule>Implicit usings are enabled.</rule>
      <rule>XML documentation comments are required for public and protected members.</rule>
    </language_framework>

    <style_rules>
      <rule>Use `PascalCase` for public/protected members and `_camelCase` for private fields.</rule>
      <rule>Prefer `var` when the type is obvious from the right-hand side.</rule>
      <rule>Keep methods short and extract complex logic into private helpers.</rule>
      <rule>Mark fields `readonly` when not reassigned after construction.</rule>
      <rule>Dispose ONNX and native resources using `using` blocks or `finally` cleanup.</rule>
      <rule>Follow existing pipeline pattern: `PreProcessWithContext` and `PostProcessWithContext`.</rule>
    </style_rules>

    <error_handling>
      <rule>Throw `ArgumentNullException` with `nameof(...)` for null arguments.</rule>
      <rule>Throw `InvalidOperationException` for invalid engine state.</rule>
      <rule>Log non-critical failures as warnings (for example warmup or GPU fallback).</rule>
      <rule>Use `ILogger`; do not use `Console.WriteLine`.</rule>
    </error_handling>
  </coding_conventions>

  <build_and_ci>
    <local_commands>
      <command>`./scripts/agent-self-check.sh`</command>
      <command>`./scripts/agent-self-check.sh --fix`</command>
      <command>`trunk check --all`</command>
      <command>`trunk check --all --fix`</command>
      <command>`dotnet build`</command>
      <command>`dotnet test`</command>
      <command>`dotnet test --collect:"XPlat Code Coverage"`</command>
    </local_commands>

    <agent_readiness>
      <rule>After any non-trivial AI-authored change, run lint/build/test before handoff.</rule>
      <rule>Preferred flow is one command: `./scripts/agent-self-check.sh`.</rule>
      <rule>If auto-fixes are expected, run `./scripts/agent-self-check.sh --fix`.</rule>
      <rule>Do not finalize while lint/build/test failures remain.</rule>
      <rule>Summarize executed commands and outcomes in the final agent response.</rule>
    </agent_readiness>

    <ci_workflows>
      <workflow>
        <name>Build and Test</name>
        <file>.github/workflows/build-test.yml</file>
        <trigger>Push/PR on `main` and `development`</trigger>
        <description>Builds and tests on Ubuntu, Windows, and macOS; uploads coverage.</description>
      </workflow>
      <workflow>
        <name>Publish NuGet</name>
        <file>.github/workflows/publish.yml</file>
        <trigger>Push to `release`</trigger>
        <description>Packs `InferenceEngine.Core` and publishes packages.</description>
      </workflow>
      <shared_action>.github/actions/build-and-test/action.yml</shared_action>
    </ci_workflows>

    <workflow_monitoring>
      <rule>Whenever committing and pushing new code, monitor CI workflows to successful completion.</rule>
      <rule>If any workflow fails, inspect logs, fix the issue, and push follow-up commits until workflows pass.</rule>
      <command>`gh run list --limit 10 --branch &lt;branch-name&gt;`</command>
      <command>`gh run watch &lt;run-id&gt; --exit-status`</command>
      <command>`gh run view &lt;run-id&gt; --log-failed`</command>
    </workflow_monitoring>

    <target_platforms>
      <item>win-x64</item>
      <item>win-arm64</item>
      <item>linux-x64</item>
      <item>linux-arm64</item>
      <item>osx-x64</item>
      <item>osx-arm64</item>
    </target_platforms>
  </build_and_ci>

  <testing_guidance>
    <rule>Use xUnit with `Xunit.SkippableFact` for conditional/environment-specific tests.</rule>
    <rule>Use `Skip.If(...)` for GPU/network/platform constrained tests.</rule>
    <rule>Name tests as `MethodName_Scenario_ExpectedResult`.</rule>
    <rule>When adding or changing behavior, add or update tests in `InferenceEngine.Tests/`.</rule>

    <coverage_policy>
      <rule>Add test coverage for all new code paths introduced by a change.</rule>
      <rule>Coverage baseline must not decrease when additions are made.</rule>
      <rule>If coverage drops, add tests until baseline is restored or exceeded before handoff.</rule>
      <command>`dotnet test --collect:"XPlat Code Coverage"`</command>
    </coverage_policy>
  </testing_guidance>

  <repository_branch_strategy>
    <branch_flow>
      <step>feature/&lt;name&gt; or personal branch (for example `nam20485`)</step>
      <step>development</step>
      <step>main</step>
      <step>release</step>
    </branch_flow>

    <branch_naming>
      <rule>Personal branch: `&lt;username&gt;`</rule>
      <rule>Feature branch: `feature/&lt;descriptive-name&gt;`</rule>
      <rule>Sub-feature branch: `feature-&lt;base&gt;/&lt;subject&gt;`</rule>
    </branch_naming>

    <pull_request_rules>
      <rule>PRs are required for merges into `development`, `main`, and `release`.</rule>
      <rule>Feature and personal branches target `development`.</rule>
      <rule>CI must pass and review comments must be resolved before merge.</rule>
      <rule>Automated agent review comments are treated as blocking.</rule>
    </pull_request_rules>

    <merge_workflow>
      <step order="1">Branch from `development` (or use personal branch for experimentation).</step>
      <step order="2">Commit, push, and open a PR targeting `development`.</step>
      <step order="3">Address review feedback and merge when approved.</step>
      <step order="4">Promote `development` to `main` via PR.</step>
      <step order="5">Promote `main` to `release` for production package publishing.</step>
    </merge_workflow>
  </repository_branch_strategy>

  <agent_integration_patterns>
    <vision_agent_example>
      <![CDATA[
public class VisionAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _detector;

    public async Task<AgentResponse> ProcessImageAsync(byte[] image)
    {
        var detections = await _detector.PredictAsync(image);
        return new AgentResponse { Objects = detections };
    }
}
      ]]>
    </vision_agent_example>

    <multimodal_agent_example>
      <![CDATA[
public class MultiModalAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _vision;
    private readonly IInferenceEngine<string, string> _nlp;

    public async Task<string> AnalyzeSceneAsync(byte[] image, string query)
    {
        var objects = await _vision.PredictAsync(image);
        var context = $"Scene contains: {string.Join(", ", objects.Select(o => o.Label))}";
        return await _nlp.PredictAsync($"{context}\nQuery: {query}");
    }
}
      ]]>
    </multimodal_agent_example>

    <custom_engine_checklist>
      <step order="1">Inherit from `BaseInferenceEngine&lt;TInput, TOutput&gt;`.</step>
      <step order="2">Override `ModelName` and optionally `ModelDownloadUrls` or `ModelRegistry`.</step>
      <step order="3">Implement `PreProcessWithContext()` for input-to-tensor conversion.</step>
      <step order="4">Implement `PostProcessWithContext()` for output decoding.</step>
      <step order="5">Provide constructor with `InferenceEngineOptions` and `ILogger&lt;T&gt;`.</step>
      <step order="6">Add tests in `InferenceEngine.Tests/`.</step>
    </custom_engine_checklist>

    <agent_lifecycle>
      <phase>
        <name>Init</name>
        <api>`engine.LoadAsync()`</api>
        <notes>Downloads model (if needed), builds ONNX session, and runs warmup.</notes>
      </phase>
      <phase>
        <name>Predict</name>
        <api>`engine.PredictAsync(input)`</api>
        <notes>Thread-safe; auto-loads if needed.</notes>
      </phase>
      <phase>
        <name>Cleanup</name>
        <api>`engine.Dispose()`</api>
        <notes>Releases ONNX session and native resources.</notes>
      </phase>
    </agent_lifecycle>

    <performance_tips>
      <item>Set `WarmupOnLoad = true` to reduce cold-start latency.</item>
      <item>Enable GPU acceleration for throughput-sensitive workloads.</item>
      <item>Dispose engines when the host agent shuts down.</item>
      <item>Consider batching for high-throughput scenarios.</item>
    </performance_tips>
  </agent_integration_patterns>

  <common_tasks_for_agents>
    <task>
      <name>Adding a model to the registry</name>
      <instruction>Edit `InferenceEngine.Core/models.json`.</instruction>
      <example><![CDATA[
{
  "name": "new-model.onnx",
  "description": "Description of the model",
  "source": "Source organization",
  "urls": [
    "https://primary-url/new-model.onnx",
    "https://fallback-url/new-model.onnx"
  ]
}
      ]]></example>
    </task>

    <task>
      <name>Adding a new engine</name>
      <step order="1">Create `InferenceEngine.Examples/NewEngine.cs` deriving from `BaseInferenceEngine&lt;TInput, TOutput&gt;`.</step>
      <step order="2">Create `InferenceEngine.Tests/NewEngineTests.cs`.</step>
      <step order="3">If model is public, add an entry to `models.json`.</step>
    </task>

    <task>
      <name>Adding a NuGet dependency</name>
      <step order="1">Add `&lt;PackageReference&gt;` to the relevant `.csproj`.</step>
      <step order="2">Prefer `Microsoft.Extensions.*` abstractions when applicable.</step>
      <step order="3">Use conditional `&lt;ItemGroup&gt;` for platform-specific packages where needed.</step>
    </task>
  </common_tasks_for_agents>

  <validation_before_handoff>
    <step order="1">Run focused tests for touched modules and run full `dotnet test` when behavior changes are broad.</step>
    <step order="2">Run lint/build/test checks (`./scripts/agent-self-check.sh` preferred).</step>
    <step order="3">For new or changed code paths, run coverage collection and confirm no baseline regression.</step>
    <step order="4">After pushing commits, monitor workflow runs and fix failures until all required workflows succeed.</step>
    <step order="5">
      Summarize in handoff:
      - What changed
      - What was validated
      - Coverage outcome versus baseline
      - Any remaining risk (especially credential/platform-specific paths)
    </step>
  </validation_before_handoff>

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
