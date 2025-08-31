---
name: wrap
description: Complete current work, create branch, commit, push, and create PR
model: claude-3-5-sonnet-latest
---

# Wrap Up Work Command

Complete all current work, create an appropriate branch, commit changes, push to GitHub, and create a pull request.

## Steps

1. **Analyze current changes** to determine branch type (feature/bugfix/docs/refactor)
2. **Create descriptive branch name** based on the work done
3. **Stage and commit changes** with detailed commit message
4. **Push branch** to remote repository
5. **Create pull request** with comprehensive description
6. **Switch back to main branch** for clean workspace
7. **Return PR URL** for review

## Process

### 1. Determine Branch Type
- Review changes using `git status` and `git diff`
- Categorize work:
  - `feature/` - New functionality added
  - `bugfix/` - Bug fixes
  - `docs/` - Documentation updates
  - `refactor/` - Code refactoring
  - `chore/` - Maintenance tasks
  - `test/` - Test additions/updates

### 2. Generate Branch Name
- Format: `type/brief-description`
- Example: `feature/token-usage-analysis`
- Keep it concise but descriptive

### 3. Commit Strategy
- Review all changes thoroughly
- Group related changes
- Write clear, descriptive commit messages
- Follow conventional commit format when possible

### 4. PR Description Template
```markdown
## Summary
[Brief description of changes]

## Changes Made
- [List of specific changes]
- [Include file modifications]

## Testing
- [How changes were tested]
- [Any test commands run]

## Related Issues
- Fixes #[issue number] (if applicable)

## Checklist
- [ ] Code follows project conventions
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if needed)
- [ ] No sensitive information exposed
```

### 5. Validation
- Ensure no secrets or sensitive data in commits
- Verify branch protection rules are followed
- Check that CI/CD will run properly

## Example Usage

```
User: wrap

Assistant will:
1. Analyze current work
2. Create branch like `feature/mcp-integration`
3. Commit with message: "feat: Add MCP server integration for enhanced tool capabilities"
4. Push to GitHub
5. Create PR with detailed description
6. Return: "Pull request created: https://github.com/user/repo/pull/123"
```

## Notes
- Always ensure work is complete before wrapping
- Review changes carefully before committing
- Include relevant context in PR description
- Tag relevant reviewers if known

## Implementation Requirements
- **Always push branch**: Use `git push -u origin <branch-name>` to push and track the branch
- **Switch to main**: Always end with `git checkout main` to return to clean main branch
- **Clean workspace**: Leave the user on main branch ready for next work