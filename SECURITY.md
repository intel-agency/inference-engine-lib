# Security Policy

## Reporting a vulnerability

Please use **GitHub's private vulnerability reporting**: open [**Report a vulnerability**](https://github.com/intel-agency/inference-engine-lib/security/advisories/new) (in the repository's [Security tab](https://github.com/intel-agency/inference-engine-lib/security)). This keeps the report private between you and the maintainers.

Please do **not** open a public issue, discussion, or pull request for an unpatched vulnerability.

Include what you can of:

- The exact `InferenceEngine.Core` package version
- Operating system and .NET RID (`win-x64`, `linux-x64`, `osx-arm64`, ...)
- Execution provider in use (`DirectML`, `CoreML`, `MIGraphX`, or `CPU`) and GPU/driver details
- Engine type (standard, streaming, or text engine) and a minimal reproduction (model + code)
- The impact you observed

We will acknowledge reports and keep you informed of progress toward a fix and release.

## Supported versions

| Package version | License | Distribution | Supported |
| :--- | :--- | :--- | :--- |
| `1.2.x` | MIT | nuget.org (public) | :white_check_mark: Supported |
| `1.1.x` and earlier | AGPL-3.0-or-later | Private GitHub Packages | :x: Unsupported — private-era releases; upgrade |

Security fixes land on the supported `1.2.x` line only. The `1.1.x` line was distributed from a private GitHub Packages feed during the project's private era and receives no security updates — upgrade to the current public line before reporting. See [Supported versions](#supported-versions) above for the full support table.

To reproduce a finding against a supported version locally, see [CONTRIBUTING.md](CONTRIBUTING.md) for build and test instructions.
