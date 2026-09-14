# fin-skills documentation

[Research workflow](RESEARCH_WORKFLOW.md) | [API compatibility](API_COMPATIBILITY.md)

The library combines importable finance skills, research checks, source adapters and public
information collection. The current source contains 127 skills, 33 guards and 50 JSON/MCP tools.

| Document | Purpose |
|---|---|
| [Collection guide](COLLECTION.md) | Real news/disclosure sources, Python/CLI/MCP setup, polling and retries |
| [Algorithm guide](ALGORITHMS.md) | Algorithm registry, automatic selection/execution, temporal comparison and custom adapters |
| [Completion audit](COMPLETION_AUDIT.md) | Requirements, measured checks and remaining account prerequisites |
| [English README](../README.md) | Installation, architecture and generated skill catalog |
| [Chinese overview](../README_ZH.md) | Chinese introduction and earlier skill explanations |
| [Python API reference](https://howard-lynn-ye.github.io/fin-skills/) | Importable modules and function signatures |
| [Examples](../examples/) | Executable research workflows and collection watchlist |
| [User manual](USER_MANUAL_ZH.md) | Historical external integration example; its datasets are not bundled |
| [Operations guide](OPERATIONS_GUIDE_ZH.md) | Historical external-host operations; adapt paths to your checkout |
| [CI and publishing](../ci/README.md) | Active checks and distribution prerequisites |

Research guards run locally. Data adapters and collectors make network requests only when
explicitly called. Importing the package does not start a background service.
