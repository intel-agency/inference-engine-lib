---
description: Unified AI agent instruction set for this repository
scope: global
role: System Orchestrator and Project Inference Specialist
---

<instructions>
  <overview>
    This document consolidates repository-level agent guidance, canonical instruction loading rules, tool-usage policy, and project memory context for the Inference Engine Library.
    Use it as a single operational reference instead of reading multiple instruction files independently.
  </overview>

  <execution_environment>
    <shell_preference>
      Assume command examples should target Microsoft PowerShell (MS PWSH / pwsh) by default unless the active environment requires another shell.
    </shell_preference>
    <platform_expectations>
      Project runtime behavior is cross-platform (Windows, macOS, Linux), with acceleration behavior selected per operating system.
    </platform_expectations>
  </execution_environment>

  <configuration>
    <branch>main</branch>
  </configuration>

  <instruction_source>
    <canonical_repository>
      <name>nam20485/agent-instructions</name>
      <url_template>https://github.com/nam20485/agent-instructions/tree/{branch}</url_template>
      <canonical_module_directory>ai_instruction_modules</canonical_module_directory>
      <canonical_module_directory_alias>ai_instrucitons_modules</canonical_module_directory_alias>
      <branch>{branch}</branch>
    </canonical_repository>
    <source_of_truth>
      Remote modules in the canonical repository are authoritative for dynamic workflows and workflow assignments.
    </source_of_truth>
  </instruction_source>

  <module_registry>
    <module type="core" required="true">
      <name>Core Instructions</name>
      <remote_path>ai_instruction_modules/ai-core-instructions.md</remote_path>
      <purpose>Defines baseline agent behavior and constraints.</purpose>
    </module>
    <module type="local" required="true">
      <name>Local AI Instructions</name>
      <path>../local_ai_instruction_modules</path>
      <purpose>Provides workspace-specific overlays.</purpose>
    </module>
    <module type="dynamic workflow" required="true">
      <name>Dynamic Workflow Orchestration</name>
      <path>../local_ai_instruction_modules/ai-dynamic-workflows.md</path>
      <purpose>Specifies how workflow definitions are resolved from canonical remote sources.</purpose>
    </module>
    <module type="workflow assignment" required="true">
      <name>Workflow Assignments</name>
      <path>../local_ai_instruction_modules/ai-workflow-assignments.md</path>
      <purpose>Maps short IDs to currently active workflows.</purpose>
    </module>
    <module type="optional">
      <name>Terminal Commands</name>
      <path>../local_ai_instruction_modules/ai-terminal-commands.md</path>
      <purpose>Reference for CLI and terminal operation patterns.</purpose>
    </module>
  </module_registry>

  <loading_protocol>
    <rule id="branch_resolution">
      Read the active branch from the configuration section and apply it to all placeholder-based links.
      If not set, use `main`.
    </rule>
    <rule id="raw_content_only">
      Retrieve remote instruction files through `raw.githubusercontent.com` URLs rather than GitHub UI pages.
    </rule>
    <algorithm name="github_ui_to_raw_translation">
      <step>Start from the configured `&lt;branch&gt;` value.</step>
      <step>Take a URL in `github.com/.../blob/{branch}/...` format.</step>
      <step>Switch host from `github.com` to `raw.githubusercontent.com`.</step>
      <step>Remove `blob/` from the path.</step>
      <step>Resolve `{branch}` with the configured branch name.</step>
    </algorithm>
  </loading_protocol>

  <policy name="single_source_of_truth">
    <rule>Do not derive dynamic workflow steps from stale local mirrors.</rule>
    <rule>Fetch workflow definitions and assignments from canonical remote sources at execution time.</rule>
    <rule>Treat remote updates as immediately effective.</rule>
  </policy>

  <tool_use_instructions>
    <instruction id="microsoft_docs_tools">
      <applyTo>Microsoft platform and .NET ecosystem work</applyTo>
      <tools>
        <tool>microsoft_docs_search</tool>
        <tool>microsoft_docs_fetch</tool>
        <tool>microsoft_code_sample_search</tool>
      </tools>
      <guidance>
        For C#, F#, ASP.NET Core, Microsoft.Extensions, NuGet, EF, and runtime-level questions, check official Microsoft documentation and samples before implementation.
        For Microsoft technology coding tasks, perform docs/sample lookup first and prioritize those findings in implementation decisions.
      </guidance>
    </instruction>

    <instruction id="sequential_thinking_default_usage">
      <applyTo>All non-trivial requests</applyTo>
      <tools>
        <tool>sequential_thinking</tool>
      </tools>
      <guidance>
        Use structured stepwise reasoning by default, excluding simple one-step tasks.
        Apply it for architecture work, unclear scopes, debugging, branching alternatives, and situations requiring plan revision as new data appears.
      </guidance>
      <capabilities>
        <item>Revise prior reasoning when new evidence changes conclusions.</item>
        <item>Branch into alternate approaches and compare outcomes.</item>
        <item>Adjust estimated step count as true scope emerges.</item>
        <item>Extend analysis when additional reasoning is required.</item>
      </capabilities>
    </instruction>

    <instruction id="memory_graph_default_usage">
      <applyTo>All non-trivial requests</applyTo>
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
        Use memory to persist reusable user, project, architecture, tooling, and workflow context.
        Read or search memory when starting complex work; write back durable facts after meaningful changes.
      </guidance>
      <knowledge_model>
        <entity>Named node with a type and factual observations.</entity>
        <relation>Directed active-voice edge between entities.</relation>
        <observation>Atomic fact statement attached to an entity.</observation>
      </knowledge_model>
      <storage_targets>
        <item>User preferences and collaboration patterns.</item>
        <item>Project conventions, architecture choices, and rationale.</item>
        <item>Recurring implementation issues and proven remedies.</item>
        <item>Team roles, tool configuration, and environment setup details.</item>
      </storage_targets>
      <tool_patterns>
        <item>`create_entities`: register people, tools, projects, or concepts.</item>
        <item>`create_relations`: link related entities with explicit verbs.</item>
        <item>`add_observations`: append atomic facts and discoveries.</item>
        <item>`search_nodes`: locate context by keywords.</item>
        <item>`open_nodes`: inspect specific entities deeply.</item>
        <item>`read_graph`: review global memory before planning/review.</item>
        <item>Delete operations: remove stale or incorrect graph entries.</item>
      </tool_patterns>
    </instruction>
  </tool_use_instructions>

  <project_profile>
    <mission>
      Deliver a reusable .NET inference foundation for AI agents and AI-enabled applications without duplicating inference lifecycle plumbing.
    </mission>
    <problems_addressed>
      <item>Repeated model-session setup and teardown logic across projects.</item>
      <item>Inconsistent model loading, warmup, and prediction flows.</item>
      <item>Platform-specific acceleration handling complexity.</item>
    </problems_addressed>
    <user_outcomes>
      <item>Fast path from model artifact to production prediction flow.</item>
      <item>Consistent extension points for custom preprocessing and postprocessing.</item>
      <item>Predictable behavior across major desktop/server platforms.</item>
    </user_outcomes>
  </project_profile>

  <agent_architecture>
    <core_components>
      <component name="IInferenceEngine&lt;TInput, TOutput&gt;">Generic async inference contract.</component>
      <component name="BaseInferenceEngine&lt;TInput, TOutput&gt;">ONNX Runtime pipeline with provider selection, load, run, warmup, and disposal lifecycle.</component>
      <component name="InferenceEngineOptions">Runtime model and acceleration settings.</component>
    </core_components>
    <integration_patterns>
      <pattern name="vision_agent">
        Inject an inference engine with image bytes as input and structured detections as output.
      </pattern>
      <pattern name="multi_modal_agent">
        Compose vision inference output with NLP inference to answer scene-aware queries.
      </pattern>
    </integration_patterns>
  </agent_architecture>

  <implementation_guidelines>
    <custom_engine_requirements>
      <item>Subclass `BaseInferenceEngine&lt;TInput, TOutput&gt;`.</item>
      <item>Implement preprocess transformation via context-aware hook.</item>
      <item>Implement postprocess output mapping via context-aware hook.</item>
      <item>Own domain-specific data conversions and result shaping.</item>
    </custom_engine_requirements>
    <lifecycle>
      <phase name="initialization">Load models and warmup as part of startup where applicable.</phase>
      <phase name="processing">Serve requests through `PredictAsync` for a consistent execution path.</phase>
      <phase name="shutdown">Dispose engines and runtime resources cleanly.</phase>
    </lifecycle>
    <performance>
      <item>Enable warmup in production when latency predictability matters.</item>
      <item>Use available hardware acceleration with graceful CPU fallback.</item>
      <item>Dispose temporary tensors/resources promptly.</item>
      <item>Consider batching for throughput-heavy workloads.</item>
    </performance>
    <reference_engine>
      `InferenceEngine.Core/Examples/FaceDetectionEngine.cs` is the primary model for image scaling, confidence filtering, coordinate conversion, and non-maximum suppression handling.
    </reference_engine>
  </implementation_guidelines>

  <repository_architecture>
    <solution>inference-engine-lib.slnx</solution>
    <projects>
      <project>InferenceEngine.Core</project>
      <project>InferenceEngine.Tests</project>
    </projects>
    <system_scope>
      Core library hosts the reusable inference abstraction and ONNX execution pipeline.
      Tests validate options behavior, coordinate transforms, non-maximum suppression behavior, and integration-level model runs.
      Face detection example demonstrates concrete computer-vision specialization.
    </system_scope>
    <key_paths>
      <path>InferenceEngine.Core/IInferenceEngine.cs</path>
      <path>InferenceEngine.Core/BaseInferenceEngine.cs</path>
      <path>InferenceEngine.Core/InferenceEngineOptions.cs</path>
      <path>InferenceEngine.Core/Examples/FaceDetectionEngine.cs</path>
      <path>InferenceEngine.Tests/*.cs</path>
    </key_paths>
    <design_patterns>
      <item>Template-method pipeline in the base engine.</item>
      <item>Provider strategy chosen by OS and settings.</item>
      <item>DTO-style prediction output models in example engine.</item>
    </design_patterns>
    <critical_paths>
      <item>`LoadAsync` validates model path, creates session, captures metadata, and optionally warms up.</item>
      <item>`PredictAsync` runs preprocess, executes session, postprocesses outputs, and handles intermediate resource cleanup.</item>
    </critical_paths>
  </repository_architecture>

  <technology_profile>
    <framework>.NET 10 (`net10.0`) for core and test projects.</framework>
    <runtime_backend>ONNX Runtime (+ DirectML package for Windows acceleration scenarios).</runtime_backend>
    <image_processing>SkiaSharp for example vision preprocessing.</image_processing>
    <testing>xUnit with SkippableFact, Microsoft.NET.Test.Sdk, and coverlet collector.</testing>
    <dependencies>
      <item>Microsoft.ML.OnnxRuntime</item>
      <item>Microsoft.ML.OnnxRuntime.Extensions</item>
      <item>Microsoft.ML.OnnxRuntime.DirectML</item>
      <item>Microsoft.Extensions.Logging.Abstractions</item>
      <item>SkiaSharp</item>
      <item>xunit</item>
      <item>xunit.runner.visualstudio</item>
      <item>Xunit.SkippableFact</item>
      <item>Microsoft.NET.Test.Sdk</item>
      <item>coverlet.collector</item>
    </dependencies>
    <constraints>
      <item>Model path must resolve to a valid ONNX artifact prior to load.</item>
      <item>Provider selection is OS-aware: DirectML on Windows, CoreML on macOS, CPU fallback otherwise.</item>
      <item>Integration tests may skip when external model/image endpoints are unavailable.</item>
    </constraints>
  </technology_profile>

  <memory_bank_context>
    <status>Initialized and active.</status>
    <focus>
      Persist factual project intent, architecture, and implementation conventions for continuity across sessions.
    </focus>
    <recent_snapshot>
      <item>Core memory documents were created and aligned to current repository state.</item>
      <item>Architecture boundaries, goals, and testing strategy are now captured.</item>
    </recent_snapshot>
    <maintenance_expectations>
      <item>Refresh context after meaningful project changes.</item>
      <item>Update architecture and tech summaries when major features or dependencies change.</item>
      <item>Keep entries concise, concrete, and verifiable.</item>
    </maintenance_expectations>
    <memory_files>
      <file>.kilocode/rules/memory-bank/brief.md</file>
      <file>.kilocode/rules/memory-bank/product.md</file>
      <file>.kilocode/rules/memory-bank/context.md</file>
      <file>.kilocode/rules/memory-bank/architecture.md</file>
      <file>.kilocode/rules/memory-bank/tech.md</file>
    </memory_files>
  </memory_bank_context>
</instructions>
