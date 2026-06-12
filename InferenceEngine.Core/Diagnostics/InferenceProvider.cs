namespace InferenceEngine.Core.Diagnostics;

/// <summary>Identifies the execution provider used by a loaded inference engine.</summary>
public enum InferenceProvider
{
    /// <summary>LoadAsync has not been called yet, or the engine has been disposed.</summary>
    Unknown = 0,
    Cpu,
    MIGraphX,
    DirectML,
    CoreML,
}
