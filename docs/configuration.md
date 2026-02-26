# Client Configuration

## Claude Code

```bash
claude mcp add tropical-mcp --scope user -- \
  uv --directory /absolute/path/to/tropical-mcp run tropical-mcp
```

## Codex

`~/.codex/config.toml`

```toml
[[mcp_servers]]
name = "tropical-mcp"
command = "uv"
args = ["--directory", "/absolute/path/to/tropical-mcp", "run", "tropical-mcp"]
```
