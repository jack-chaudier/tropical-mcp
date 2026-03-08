# Security Policy

`tropical-mcp` is an evaluation server, but security still matters because it runs inside coding-agent workflows, emits telemetry, and is expected to be inspected by outside researchers.

## Report privately

If you believe you found a security issue in `tropical-mcp`, please report it privately:

- Email: `jackgaff@umich.edu`
- Subject: `tropical-mcp security report`

Please include:

- affected version or commit
- reproduction steps
- expected versus actual behavior
- potential impact
- any logs or artifacts needed to reproduce, with secrets removed

Do not open public issues for unpatched vulnerabilities.

## In scope

- anything that could cause JSON-RPC traffic or sensitive content to leak to stdout
- telemetry-path, fixture, or packaging behavior that exposes unintended local information
- dependency, build, or release issues that weaken the shipped validation surface
- accidental publication of secrets, credentials, or private research material

## Out of scope

- disagreements about research conclusions that do not create a security or privacy issue
- feature requests for unsupported host-client interception behavior
- issues in third-party clients that are outside this repository's control

## Release-day maintainer checklist

- run `uv run --extra dev pytest`
- run `uv build`
- run `./scripts/validate_installed_wheel.sh`
- run `uv run tropical-mcp-full-validate`
- run `./scripts/update_release_checksums.sh`
- confirm no secrets or machine-local paths leaked into examples, fixtures, or docs
