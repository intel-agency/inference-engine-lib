namespace InferenceEngine.Core.Diagnostics;

/// <summary>One step in the provider-selection diagnostic trail.</summary>
public sealed record ProviderDiagnosticCheck
{
    /// <summary>Stable dotted identifier (e.g. "rocm.append", "kfd.present"). Tests assert on these.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Human-readable description.</summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>Grouping for UI (e.g. "Platform", "Provider", "Fallback", "Session").</summary>
    public string Category { get; init; } = string.Empty;

    public CheckResult Result { get; init; }

    /// <summary>Detail (exception message, file path tried, etc.). Null when not informative.</summary>
    public string? Detail { get; init; }

    /// <summary>0-based ordinal of evaluation; must equal the check's index in the parent list.</summary>
    public int Order { get; init; }

    public static ProviderDiagnosticCheck Pass(string name, string description, string category, int order, string? detail = null) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Passed, Detail = detail,
    };

    public static ProviderDiagnosticCheck Fail(string name, string description, string category, int order, string detail) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Failed, Detail = detail,
    };

    public static ProviderDiagnosticCheck Skip(string name, string description, string category, int order, string reason) => new()
    {
        Name = name, Description = description, Category = category, Order = order, Result = CheckResult.Skipped, Detail = reason,
    };
}
