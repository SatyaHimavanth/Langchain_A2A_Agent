# Project setup
```bash
uv sync
```

# Start hosting calculator agent (__main__.py)
```bash
uv run .
```

# Test it using 
```bash
uv run test_client.py
```

# Auth testing (advanced tools: power, root)
Set `A2A_AUTH_TOKEN` in `.env` for authorized mode, or unset it for basic mode.
Server-side valid tokens come from `A2A_AUTH_TOKENS` (comma-separated).


# Reference links:

## langgraph agent to a2a
```bash
https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/langgraph
```

## Complete git repo
```bash
https://github.com/a2aproject/a2a-samples
```

