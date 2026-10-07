### Summary

<!-- What does this PR change, and why? Reference the issue it closes if any. -->

### Package impact

<!-- Does this change the package, its metadata, dependencies, or versioning?
     If yes, describe the change. -->

- [ ] No package / metadata / dependency / versioning change
- [ ] Package or versioning change (details above)

### Validation performed

<!-- CI run link plus local results. The standard suite excludes
     FaceDetectionEngineTests and NativeLibraryValidationTests
     (GPU/network/model-download dependent). -->

- [ ] `dotnet build` green
- [ ] Standard test suite green:
      `dotnet test --filter "FullyQualifiedName!~FaceDetectionEngineTests&FullyQualifiedName!~NativeLibraryValidationTests"`
- [ ] Coverage not regressed (baseline: 68.9% line / 54.3% branch)
- [ ] Commits are GPG-signed; no squash/rebase history
