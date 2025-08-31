#!/bin/bash

# MCP Server Manager - Installation, Update, and Health Check Script
# Purpose: Manage MCP (Model Context Protocol) servers for Claude Code
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MCP_CONFIG_DIR="$HOME/.config/claude"
MCP_SERVERS_DIR="$HOME/.local/share/claude-code/mcp-servers"
LOG_FILE="/tmp/mcp_manager_$(date +%Y%m%d_%H%M%S).log"

# MCP Server definitions
declare -A MCP_SERVERS=(
    ["serena"]="github:serena/semantic-code-retrieval"
    ["context7"]="npm:@context7/mcp-server"
    ["sequential-thinking"]="npm:@modelcontextprotocol/sequential-thinking"
    ["github-workflow-manager"]="github:workflow-manager/github-mcp"
    ["system-architect"]="npm:@mcp/system-architect"
    ["docs-sync-validator"]="github:docs-sync/validator-mcp"
)

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success") echo -e "${GREEN}✓${NC} $message" ;;
        "error") echo -e "${RED}✗${NC} $message" ;;
        "warning") echo -e "${YELLOW}⚠${NC} $message" ;;
        "info") echo -e "${BLUE}ℹ${NC} $message" ;;
        "header") echo -e "${CYAN}═══ $message ═══${NC}" ;;
    esac
    log "$status: $message"
}

# Check prerequisites
check_prerequisites() {
    print_status "header" "Checking Prerequisites"
    
    local prereqs_met=true
    
    # Check for Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        print_status "success" "Node.js installed: $node_version"
    else
        print_status "error" "Node.js not found. Please install Node.js 18+ first."
        prereqs_met=false
    fi
    
    # Check for npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version)
        print_status "success" "npm installed: $npm_version"
    else
        print_status "error" "npm not found. Please install npm."
        prereqs_met=false
    fi
    
    # Check for git
    if command -v git &> /dev/null; then
        print_status "success" "git installed"
    else
        print_status "error" "git not found. Please install git."
        prereqs_met=false
    fi
    
    # Check for Claude CLI
    if command -v claude &> /dev/null; then
        print_status "success" "Claude CLI installed"
    else
        print_status "warning" "Claude CLI not found. Some features may not work."
    fi
    
    if [ "$prereqs_met" = false DestReader, i.e. ]; then
        return 1
    fi
    return 0
}

# Create necessary directories
setup_directories() {
    print_status "info" "Setting up directories..."
    
    mkdir -p "$MCP_CONFIG_DIR"
    mkdir -p "$MCP_SERVERS_DIR"
    mkdir -p "$HOME/.cache/mcp"
    
    print_status "success" "Directories created/verified"
}

# Install a single MCP server
install_mcp_server() {
    local name=$1
    local source=$2
    local skip_on_fail=${3:-true}  # Default to skip on failure
    
    print_status "info" "Installing $name..."
    
    local install_dir="$MCP_SERVERS_DIR/$name"
    mkdir -p "$install_dir"
    
    # Track installation success
    local install_success=false
    
    if [[ "$source" == npm:* ]]; then
        # NPM package installation
        local package="${source#npm:}"
        cd "$install_dir"
        
        # Initialize package.json if not exists
        if [ ! -f package.json ]; then
            npm init -y &>> "$LOG_FILE" 2>&1
        fi
        
        # Install the package
        if npm install "$package" &>> "$LOG_FILE" 2>&1; then
            print_status "success" "$name installed via npm"
            install_success=true
        else
            print_status "error" "Failed to install $name via npm"
            if [ "$skip_on_fail" = true ]; then
                print_status "warning" "Skipping $name and continuing with next server..."
            fi
        fi
        
    elif [[ "$source" == github:* ]]; then
        # GitHub repository installation
        local repo="${source#github:}"
        local repo_url="https://github.com/$repo.git"
        
        # Clone or update repository
        if [ -d "$install_dir/.git" ]; then
            print_status "info" "$name already cloned, pulling updates..."
            cd "$install_dir"
            if git pull &>> "$LOG_FILE" 2>&1; then
                install_success=true
            else
                print_status "error" "Failed to update $name from GitHub"
            fi
        else
            if git clone "$repo_url" "$install_dir" &>> "$LOG_FILE" 2>&1; then
                cd "$install_dir"
                install_success=true
            else
                print_status "error" "Failed to clone $name from GitHub"
            fi
        fi
        
        # If clone/update succeeded, install dependencies
        if [ "$install_success" = true ]; then
            # Install dependencies if package.json exists
            if [ -f package.json ]; then
                if ! npm install &>> "$LOG_FILE" 2>&1; then
                    print_status "warning" "Failed to install npm dependencies for $name"
                    install_success=false
                fi
            fi
            
            # Build if needed
            if [ "$install_success" = true ] && [ -f package.json ] && grep -q '"build"' package.json; then
                if ! npm run build &>> "$LOG_FILE" 2>&1; then
                    print_status "warning" "Failed to build $name"
                    install_success=false
                fi
            fi
        fi
        
        if [ "$install_success" = true ]; then
            print_status "success" "$name installed from GitHub"
        elif [ "$skip_on_fail" = true ]; then
            print_status "warning" "Skipping $name and continuing with next server..."
        fi
    else
        print_status "error" "Unknown source type for $name: $source"
        if [ "$skip_on_fail" = true ]; then
            print_status "warning" "Skipping $name and continuing with next server..."
        fi
    fi
    
    # Return status
    if [ "$install_success" = true ]; then
        return 0
    else
        return 1
    fi
}

# Update MCP configuration
update_mcp_config() {
    print_status "header" "Updating MCP Configuration"
    
    local config_file="$MCP_CONFIG_DIR/mcp-config.json"
    
    # Create base configuration
    cat > "$config_file" << 'EOF'
{
  "mcpServers": {
    "serena": {
      "command": "node",
      "args": ["~/.local/share/claude-code/mcp-servers/serena/index.js"],
      "env": {},
      "capabilities": ["semantic_search", "code_edit", "file_management"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"],
      "env": {},
      "capabilities": ["documentation", "library_info"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/sequential-thinking"],
      "env": {},
      "capabilities": ["reasoning", "planning"]
    },
    "github-workflow-manager": {
      "command": "node",
      "args": ["~/.local/share/claude-code/mcp-servers/github-workflow-manager/dist/index.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      },
      "capabilities": ["git", "github", "workflow"]
    },
    "system-architect": {
      "command": "npx",
      "args": ["-y", "@mcp/system-architect"],
      "env": {},
      "capabilities": ["architecture", "design", "planning"]
    },
    "docs-sync-validator": {
      "command": "node",
      "args": ["~/.local/share/claude-code/mcp-servers/docs-sync-validator/index.js"],
      "env": {},
      "capabilities": ["documentation", "validation", "sync"]
    }
  },
  "globalSettings": {
    "timeout": 30000,
    "retryCount": 3,
    "logLevel": "info"
  }
}
EOF
    
    # Expand home directory in paths
    sed -i "s|~|$HOME|g" "$config_file"
    
    print_status "success" "Configuration updated at $config_file"
}

# Health check for a single MCP server
check_mcp_server() {
    local name=$1
    local install_dir="$MCP_SERVERS_DIR/$name"
    
    echo -n "Checking $name... "
    
    # Check if directory exists
    if [ ! -d "$install_dir" ]; then
        print_status "error" "$name not installed"
        return 1
    fi
    
    # Check for main entry point
    if [ -f "$install_dir/index.js" ] || [ -f "$install_dir/dist/index.js" ] || [ -f "$install_dir/package.json" ]; then
        # Try to verify the server can start
        cd "$install_dir"
        
        # Quick Node.js syntax check
        if [ -f index.js ]; then
            if node -c index.js &> /dev/null; then
                print_status "success" "$name is healthy"
                return 0
            else
                print_status "warning" "$name has syntax errors"
                return 1
            fi
        elif [ -f dist/index.js ]; then
            if node -c dist/index.js &> /dev/null; then
                print_status "success" "$name is healthy"
                return 0
            else
                print_status "warning" "$name has syntax errors"
                return 1
            fi
        else
            print_status "success" "$name appears configured"
            return 0
        fi
    else
        print_status "error" "$name missing entry point"
        return 1
    fi
}

# Install all MCP servers
install_all() {
    print_status "header" "Installing MCP Servers"
    
    local total_count=${#MCP_SERVERS[@]}
    local success_count=0
    local failed_count=0
    local skipped_count=0
    local failed_servers=()
    local success_servers=()
    
    for server in "${!MCP_SERVERS[@]}"; do
        if install_mcp_server "$server" "${MCP_SERVERS[$server]}" true; then
            ((success_count++))
            success_servers+=("$server")
        else
            ((failed_count++))
            failed_servers+=("$server")
        fi
    done
    
    # Print installation summary
    echo ""
    print_status "header" "Installation Summary"
    echo "Total servers: $total_count"
    echo "Successfully installed: $success_count"
    echo "Failed installations: $failed_count"
    
    if [ ${#success_servers[@]} -gt 0 ]; then
        echo ""
        print_status "success" "Successfully installed servers:"
        for server in "${success_servers[@]}"; do
            echo "  ✓ $server"
        done
    fi
    
    if [ ${#failed_servers[@]} -gt 0 ]; then
        echo ""
        print_status "warning" "Failed servers (skipped):"
        for server in "${failed_servers[@]}"; do
            echo "  ✗ $server"
        done
        echo ""
        print_status "info" "You can retry failed installations individually or run 'repair' later"
    fi
    
    if [ $success_count -eq $total_count ]; then
        print_status "success" "All servers installed successfully!"
    elif [ $success_count -gt 0 ]; then
        print_status "warning" "Partial installation complete ($success_count/$total_count succeeded)"
    else
        print_status "error" "All installations failed - check $LOG_FILE for details"
    fi
    
    # Update config even if some failed - configure the successful ones
    if [ $success_count -gt 0 ]; then
        update_mcp_config
    fi
}

# Update all MCP servers
update_all() {
    print_status "header" "Updating MCP Servers"
    
    for server in "${!MCP_SERVERS[@]}"; do
        local install_dir="$MCP_SERVERS_DIR/$server"
        
        if [ ! -d "$install_dir" ]; then
            print_status "warning" "$server not installed, skipping..."
            continue
        fi
        
        print_status "info" "Updating $server..."
        
        cd "$install_dir"
        
        # Update based on type
        if [ -d .git ]; then
            git pull &>> "$LOG_FILE"
            if [ -f package.json ]; then
                npm update &>> "$LOG_FILE"
                npm install &>> "$LOG_FILE"
            fi
        elif [ -f package.json ]; then
            npm update &>> "$LOG_FILE"
        fi
        
        print_status "success" "$server updated"
    done
    
    update_mcp_config
}

# Health check all MCP servers
health_check_all() {
    print_status "header" "MCP Server Health Check"
    
    local healthy_count=0
    local total_count=${#MCP_SERVERS[@]}
    
    for server in "${!MCP_SERVERS[@]}"; do
        if check_mcp_server "$server"; then
            ((healthy_count++))
        fi
    done
    
    echo ""
    print_status "info" "Health Summary: $healthy_count/$total_count servers healthy"
    
    # Check configuration
    if [ -f "$MCP_CONFIG_DIR/mcp-config.json" ]; then
        print_status "success" "Configuration file exists"
    else
        print_status "warning" "Configuration file missing - run install to create"
    fi
    
    # Test Claude CLI integration
    if command -v claude &> /dev/null; then
        echo ""
        print_status "info" "Testing Claude CLI integration..."
        if claude mcp list &> /dev/null; then
            print_status "success" "Claude CLI can access MCP servers"
        else
            print_status "warning" "Claude CLI cannot access MCP servers"
        fi
    fi
}

# Repair broken installations
repair() {
    print_status "header" "Repairing MCP Installations"
    
    for server in "${!MCP_SERVERS[@]}"; do
        if ! check_mcp_server "$server" &> /dev/null; then
            print_status "info" "Repairing $server..."
            install_mcp_server "$server" "${MCP_SERVERS[$server]}"
        fi
    done
    
    update_mcp_config
    print_status "success" "Repair complete"
}

# Clean installation (remove and reinstall)
clean_install() {
    print_status "header" "Clean Installation"
    
    read -p "This will remove all existing MCP servers and reinstall. Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "info" "Clean installation cancelled"
        return
    fi
    
    print_status "warning" "Removing existing installations..."
    rm -rf "$MCP_SERVERS_DIR"
    rm -f "$MCP_CONFIG_DIR/mcp-config.json"
    
    setup_directories
    install_all
}

# Show status
show_status() {
    print_status "header" "MCP Server Status"
    
    echo "Configuration Directory: $MCP_CONFIG_DIR"
    echo "Servers Directory: $MCP_SERVERS_DIR"
    echo "Log File: $LOG_FILE"
    echo ""
    
    echo "Registered Servers:"
    for server in "${!MCP_SERVERS[@]}"; do
        local install_dir="$MCP_SERVERS_DIR/$server"
        if [ -d "$install_dir" ]; then
            echo "  • $server: Installed"
        else
            echo "  • $server: Not installed"
        fi
    done
}

# Main menu
show_menu() {
    echo ""
    print_status "header" "MCP Server Manager"
    echo ""
    echo "1) Install all MCP servers"
    echo "2) Update all MCP servers"
    echo "3) Health check"
    echo "4) Repair broken installations"
    echo "5) Clean install (remove and reinstall)"
    echo "6) Show status"
    echo "7) View log file"
    echo "8) Exit"
    echo ""
}

# Main execution
main() {
    # Create log file
    touch "$LOG_FILE"
    log "MCP Manager started"
    
    # Check prerequisites first
    if ! check_prerequisites; then
        print_status "error" "Prerequisites not met. Please install missing dependencies."
        exit 1
    fi
    
    # Setup directories
    setup_directories
    
    # Handle command line arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            install) install_all ;;
            update) update_all ;;
            check|health) health_check_all ;;
            repair) repair ;;
            clean) clean_install ;;
            status) show_status ;;
            *)
                echo "Usage: $0 [install|update|check|health|repair|clean|status]"
                echo "Or run without arguments for interactive menu"
                exit 1
                ;;
        esac
    else
        # Interactive menu
        while true; do
            show_menu
            read -p "Enter choice [1-8]: " choice
            
            case $choice in
                1) install_all ;;
                2) update_all ;;
                3) health_check_all ;;
                4) repair ;;
                5) clean_install ;;
                6) show_status ;;
                7) less "$LOG_FILE" ;;
                8) 
                    print_status "info" "Exiting..."
                    break
                    ;;
                *)
                    print_status "error" "Invalid choice"
                    ;;
            esac
            
            echo ""
            read -p "Press Enter to continue..."
        done
    fi
    
    log "MCP Manager completed"
}

# Run main function
main "$@"