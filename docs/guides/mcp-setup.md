# MCP Server Setup Guide

## Overview

MCP (Model Context Protocol) servers provide specialized capabilities to Claude Code, enabling advanced features like semantic code search, GitHub workflow management, system architecture design, and more.

## Quick Start

### Prerequisites

Ensure you have the following installed:
- Node.js 18+ 
- npm
- git
- Claude CLI (optional but recommended)

### Installation Methods

#### Method 1: Bash Script (Linux/macOS)

```bash
# Navigate to scripts directory
cd /path/to/doc-fun/scripts

# Run the manager
./mcp_manager.sh

# Or run directly with command
./mcp_manager.sh install    # Install all servers
./mcp_manager.sh health     # Check health
./mcp_manager.sh update     # Update all servers
```

#### Method 2: Python Script (Cross-platform)

```bash
# Run the Python manager
python3 scripts/mcp_manager.py

# Or with specific commands
python3 scripts/mcp_manager.py install
python3 scripts/mcp_manager.py check
python3 scripts/mcp_manager.py update
```

## MCP Servers Included

### 1. **serena**
- **Purpose**: Semantic code retrieval and editing
- **Capabilities**: 
  - Find symbols and patterns in code
  - Semantic search across codebase
  - Smart file management

### 2. **context7**
- **Purpose**: Third-party library documentation
- **Capabilities**:
  - Real-time documentation lookup
  - API reference retrieval
  - Version-specific information

### 3. **sequential-thinking**
- **Purpose**: Systematic reasoning and planning
- **Capabilities**:
  - Step-by-step problem solving
  - Decision tree exploration
  - Logical reasoning chains

### 4. **github-workflow-manager**
- **Purpose**: GitHub and git operations
- **Capabilities**:
  - PR creation and management
  - Branch operations
  - CI/CD pipeline configuration
- **Note**: Requires `GITHUB_TOKEN` environment variable

### 5. **system-architect**
- **Purpose**: System design and architecture
- **Capabilities**:
  - Architecture diagrams
  - Design patterns
  - Technology stack recommendations

### 6. **docs-sync-validator**
- **Purpose**: Documentation validation and synchronization
- **Capabilities**:
  - Documentation consistency checks
  - Cross-reference validation
  - Format standardization

## Configuration

The MCP configuration is stored at `~/.config/claude/mcp-config.json`. The manager scripts automatically create and update this file.

### Manual Configuration

If needed, you can manually edit the configuration:

```json
{
  "mcpServers": {
    "serena": {
      "command": "node",
      "args": ["~/.local/share/claude-code/mcp-servers/serena/index.js"],
      "env": {},
      "capabilities": ["semantic_search", "code_edit", "file_management"]
    }
    // ... other servers
  },
  "globalSettings": {
    "timeout": 30000,
    "retryCount": 3,
    "logLevel": "info"
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Prerequisites Not Met
```bash
# Install Node.js (using nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Install git
sudo apt-get install git  # Debian/Ubuntu
brew install git           # macOS
```

#### 2. Permission Denied
```bash
# Make scripts executable
chmod +x scripts/mcp_manager.sh
chmod +x scripts/mcp_manager.py
```

#### 3. MCP Servers Not Detected by Claude
```bash
# Run health check
./scripts/mcp_manager.sh health

# Repair installations
./scripts/mcp_manager.sh repair

# Or do a clean install
./scripts/mcp_manager.sh clean
```

#### 4. GitHub Token Issues
```bash
# Set GitHub token for github-workflow-manager
export GITHUB_TOKEN="your-github-token"

# Add to shell profile for persistence
echo 'export GITHUB_TOKEN="your-github-token"' >> ~/.bashrc
```

### Health Check Output

A healthy installation should show:
```
✓ serena is healthy
✓ context7 is healthy
✓ sequential-thinking is healthy
✓ github-workflow-manager is healthy
✓ system-architect is healthy
✓ docs-sync-validator is healthy

ℹ Health Summary: 6/6 servers healthy
✓ Configuration file exists
✓ Claude CLI can access MCP servers
```

## Manager Script Commands

### Install
Installs all MCP servers and creates configuration
```bash
./mcp_manager.sh install
```

### Update
Updates all installed servers to latest versions
```bash
./mcp_manager.sh update
```

### Health Check
Verifies all servers are properly installed and configured
```bash
./mcp_manager.sh health
```

### Repair
Attempts to fix broken installations
```bash
./mcp_manager.sh repair
```

### Clean Install
Removes all servers and performs fresh installation
```bash
./mcp_manager.sh clean
```

### Status
Shows current installation status
```bash
./mcp_manager.sh status
```

## Logs

Logs are stored in `/tmp/mcp_manager_[timestamp].log` for debugging purposes.

To view the latest log:
```bash
# From interactive menu, choose option 7
# Or manually:
ls -la /tmp/mcp_manager_*.log
tail -f /tmp/mcp_manager_latest.log
```

## Integration with Claude Code

Once installed, the MCP servers should be automatically available in Claude Code. You can verify by:

1. Opening Claude Code
2. Running `/mcp` command
3. You should see all configured servers

If servers aren't showing:
1. Restart Claude Code
2. Run `claude mcp list` in terminal
3. Check the configuration file at `~/.config/claude/mcp-config.json`

## Advanced Usage

### Installing Specific Server
```python
python3 scripts/mcp_manager.py install --server serena
```

### Verbose Mode
```bash
./mcp_manager.sh -v install  # Bash
python3 scripts/mcp_manager.py --verbose install  # Python
```

### Custom Configuration
You can add custom MCP servers by editing the script's `MCP_SERVERS` dictionary or configuration.

## Support

For issues specific to:
- **MCP Manager Scripts**: Check logs in `/tmp/mcp_manager_*.log`
- **Claude Code**: Run `claude doctor` for diagnostics
- **Individual Servers**: Check their respective GitHub repositories

---

*Last Updated: 2024-01-27*