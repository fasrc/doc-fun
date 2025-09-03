---
name: goat
description: General work orchestration with automatic tool selection and branch management
---

# GOAT Command - General Orchestration and Task Management

## Branch Management Requirements
- **Always check current branch** before starting work
- **If on main branch**: Automatically create a new feature branch based on the task
- **Branch naming convention**: `type/brief-description` where type is:
  - `feature/` - New functionality
  - `bugfix/` - Bug fixes  
  - `docs/` - Documentation updates
  - `refactor/` - Code refactoring
  - `chore/` - Maintenance tasks
  - `test/` - Test updates

## Tool Selection Guidelines
Always use:
- **serena** for semantic code retrieval and editing tools
- **context7 MCP** for programmatically searching and fetching documentation on third party libraries
- **sequential-thinking** for any decision making
- **github-workflow-manager** for gh and github flow
- **system-architect** for system architecture design, patterns, and high-level technical decisions
- **docs-sync-validator** for documentation

## Context7 MCP Integration
When working with third-party libraries or frameworks:
1. **Search for libraries** using `mcp__context7__resolve-library-id` to find Context7-compatible library IDs
2. **Fetch documentation** using `mcp__context7__get-library-docs` with the resolved library ID
3. **Use focused queries** with the `topic` parameter to get relevant documentation sections
4. **Always resolve library IDs first** unless the user provides explicit Context7-compatible library IDs

## Process
1. **Read CLAUDE.md** for project context before doing anything
2. **Check current git branch**
3. **If on main**: Create appropriate feature branch before starting work
4. **Execute the requested task** using appropriate tools
5. **Follow project conventions** and best practices

## Implementation Requirements
- Must check `git branch --show-current` before starting work
- If result is `main`, must create new branch with `git checkout -b type/description`
- Branch name should be descriptive and follow conventional format
- Always inform user of branch creation

## Task Execution
Execute the user's request: #$ARGUMENTS
