#!/usr/bin/env python3
"""
MCP Server Manager - Installation, Update, and Health Check Tool
Purpose: Manage MCP (Model Context Protocol) servers for Claude Code
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Color:
    """Terminal color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class Status(Enum):
    """Status types for colored output"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HEADER = "header"


class MCPServerManager:
    """Manages MCP server installations and configurations"""
    
    # MCP Server definitions
    MCP_SERVERS = {
        "serena": {
            "source": "github:serena/semantic-code-retrieval",
            "type": "github",
            "repo": "serena/semantic-code-retrieval",
            "entry": "index.js",
            "capabilities": ["semantic_search", "code_edit", "file_management"]
        },
        "context7": {
            "source": "npm:@context7/mcp-server",
            "type": "npm",
            "package": "@context7/mcp-server",
            "capabilities": ["documentation", "library_info"]
        },
        "sequential-thinking": {
            "source": "npm:@modelcontextprotocol/sequential-thinking",
            "type": "npm",
            "package": "@modelcontextprotocol/sequential-thinking",
            "capabilities": ["reasoning", "planning"]
        },
        "github-workflow-manager": {
            "source": "github:workflow-manager/github-mcp",
            "type": "github",
            "repo": "workflow-manager/github-mcp",
            "entry": "dist/index.js",
            "capabilities": ["git", "github", "workflow"],
            "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
        },
        "system-architect": {
            "source": "npm:@mcp/system-architect",
            "type": "npm",
            "package": "@mcp/system-architect",
            "capabilities": ["architecture", "design", "planning"]
        },
        "docs-sync-validator": {
            "source": "github:docs-sync/validator-mcp",
            "type": "github",
            "repo": "docs-sync/validator-mcp",
            "entry": "index.js",
            "capabilities": ["documentation", "validation", "sync"]
        }
    }
    
    def __init__(self, verbose: bool = False):
        """Initialize MCP Server Manager"""
        self.home = Path.home()
        self.config_dir = self.home / ".config" / "claude"
        self.servers_dir = self.home / ".local" / "share" / "claude-code" / "mcp-servers"
        self.cache_dir = self.home / ".cache" / "mcp"
        
        # Setup logging
        self.setup_logging(verbose)
        
        # Ensure directories exist
        self.setup_directories()
    
    def setup_logging(self, verbose: bool):
        """Setup logging configuration"""
        log_level = logging.DEBUG if verbose else logging.INFO
        log_file = f"/tmp/mcp_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
    
    def print_status(self, status: Status, message: str):
        """Print colored status message"""
        icons = {
            Status.SUCCESS: f"{Color.GREEN}✓{Color.NC}",
            Status.ERROR: f"{Color.RED}✗{Color.NC}",
            Status.WARNING: f"{Color.YELLOW}⚠{Color.NC}",
            Status.INFO: f"{Color.BLUE}ℹ{Color.NC}",
            Status.HEADER: f"{Color.CYAN}═══{Color.NC}"
        }
        
        if status == Status.HEADER:
            print(f"\n{icons[status]} {message} {icons[status]}")
        else:
            print(f"{icons[status]} {message}")
        
        self.logger.info(f"{status.value}: {message}")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [self.config_dir, self.servers_dir, self.cache_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.print_status(Status.SUCCESS, "Directories created/verified")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        self.print_status(Status.HEADER, "Checking Prerequisites")
        
        prerequisites = {
            "node": "Node.js",
            "npm": "npm",
            "git": "git"
        }
        
        all_met = True
        
        for cmd, name in prerequisites.items():
            if shutil.which(cmd):
                try:
                    if cmd == "node":
                        version = subprocess.check_output(
                            [cmd, "--version"], 
                            stderr=subprocess.DEVNULL
                        ).decode().strip()
                        self.print_status(Status.SUCCESS, f"{name} installed: {version}")
                    else:
                        self.print_status(Status.SUCCESS, f"{name} installed")
                except Exception:
                    self.print_status(Status.WARNING, f"{name} found but version check failed")
            else:
                self.print_status(Status.ERROR, f"{name} not found. Please install {name} first.")
                all_met = False
        
        # Check Claude CLI (optional)
        if shutil.which("claude"):
            self.print_status(Status.SUCCESS, "Claude CLI installed")
        else:
            self.print_status(Status.WARNING, "Claude CLI not found. Some features may not work.")
        
        return all_met
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}")
            self.logger.error(f"Error: {e.stderr}")
            return False, e.stderr
        except Exception as e:
            self.logger.error(f"Command exception: {e}")
            return False, str(e)
    
    def install_npm_server(self, name: str, package: str, install_dir: Path) -> bool:
        """Install an MCP server from npm"""
        self.print_status(Status.INFO, f"Installing {name} via npm...")
        
        # Create package.json if not exists
        package_json = install_dir / "package.json"
        if not package_json.exists():
            success, _ = self.run_command(["npm", "init", "-y"], cwd=install_dir)
            if not success:
                return False
        
        # Install the package
        success, _ = self.run_command(["npm", "install", package], cwd=install_dir)
        
        if success:
            self.print_status(Status.SUCCESS, f"{name} installed via npm")
        else:
            self.print_status(Status.ERROR, f"Failed to install {name} via npm")
        
        return success
    
    def install_github_server(self, name: str, repo: str, install_dir: Path) -> bool:
        """Install an MCP server from GitHub"""
        repo_url = f"https://github.com/{repo}.git"
        
        if install_dir.exists() and (install_dir / ".git").exists():
            self.print_status(Status.INFO, f"{name} already cloned, pulling updates...")
            success, _ = self.run_command(["git", "pull"], cwd=install_dir)
        else:
            self.print_status(Status.INFO, f"Cloning {name} from GitHub...")
            success, _ = self.run_command(["git", "clone", repo_url, str(install_dir)])
        
        if not success:
            self.print_status(Status.ERROR, f"Failed to clone/update {name}")
            return False
        
        # Install dependencies if package.json exists
        package_json = install_dir / "package.json"
        if package_json.exists():
            self.print_status(Status.INFO, f"Installing dependencies for {name}...")
            success, _ = self.run_command(["npm", "install"], cwd=install_dir)
            
            # Build if needed
            with open(package_json) as f:
                pkg_data = json.load(f)
                if "scripts" in pkg_data and "build" in pkg_data["scripts"]:
                    self.print_status(Status.INFO, f"Building {name}...")
                    success, _ = self.run_command(["npm", "run", "build"], cwd=install_dir)
        
        if success:
            self.print_status(Status.SUCCESS, f"{name} installed from GitHub")
        
        return success
    
    def install_server(self, name: str, skip_on_fail: bool = True) -> bool:
        """Install a single MCP server
        
        Args:
            name: Server name to install
            skip_on_fail: If True, continue with next server on failure
            
        Returns:
            True if installation succeeded, False otherwise
        """
        if name not in self.MCP_SERVERS:
            self.print_status(Status.ERROR, f"Unknown server: {name}")
            if skip_on_fail:
                self.print_status(Status.WARNING, f"Skipping {name} and continuing...")
            return False
        
        server_info = self.MCP_SERVERS[name]
        install_dir = self.servers_dir / name
        install_dir.mkdir(parents=True, exist_ok=True)
        
        success = False
        
        try:
            if server_info["type"] == "npm":
                success = self.install_npm_server(name, server_info["package"], install_dir)
            elif server_info["type"] == "github":
                success = self.install_github_server(name, server_info["repo"], install_dir)
            else:
                self.print_status(Status.ERROR, f"Unknown server type: {server_info['type']}")
        except Exception as e:
            self.print_status(Status.ERROR, f"Exception installing {name}: {str(e)}")
            self.logger.exception(f"Exception details for {name}:")
        
        if not success and skip_on_fail:
            self.print_status(Status.WARNING, f"Skipping {name} and continuing with next server...")
        
        return success
    
    def install_all(self, skip_on_fail: bool = True):
        """Install all MCP servers
        
        Args:
            skip_on_fail: If True, continue with next server on failure
        """
        self.print_status(Status.HEADER, "Installing MCP Servers")
        
        total = len(self.MCP_SERVERS)
        success_servers = []
        failed_servers = []
        
        for name in self.MCP_SERVERS:
            if self.install_server(name, skip_on_fail):
                success_servers.append(name)
            else:
                failed_servers.append(name)
        
        # Print installation summary
        print("\n" + "="*50)
        self.print_status(Status.HEADER, "Installation Summary")
        print(f"Total servers: {total}")
        print(f"Successfully installed: {len(success_servers)}")
        print(f"Failed installations: {len(failed_servers)}")
        
        if success_servers:
            print()
            self.print_status(Status.SUCCESS, "Successfully installed servers:")
            for server in success_servers:
                print(f"  ✓ {server}")
        
        if failed_servers:
            print()
            self.print_status(Status.WARNING, "Failed servers (skipped):")
            for server in failed_servers:
                print(f"  ✗ {server}")
            print()
            self.print_status(Status.INFO, "You can retry failed installations individually or run 'repair' later")
            self.print_status(Status.INFO, f"Check log file for details: {self.log_file}")
        
        # Overall status
        if len(success_servers) == total:
            self.print_status(Status.SUCCESS, "All servers installed successfully!")
        elif success_servers:
            self.print_status(Status.WARNING, f"Partial installation complete ({len(success_servers)}/{total} succeeded)")
        else:
            self.print_status(Status.ERROR, f"All installations failed - check {self.log_file} for details")
        
        # Update config even if some failed - configure the successful ones
        if success_servers:
            self.update_config(only_servers=success_servers)
    
    def update_config(self, only_servers: Optional[List[str]] = None):
        """Update MCP configuration file
        
        Args:
            only_servers: If provided, only configure these servers
        """
        self.print_status(Status.HEADER, "Updating MCP Configuration")
        
        config = {"mcpServers": {}, "globalSettings": {
            "timeout": 30000,
            "retryCount": 3,
            "logLevel": "info"
        }}
        
        # Determine which servers to configure
        servers_to_config = only_servers if only_servers else self.MCP_SERVERS.keys()
        
        for name in servers_to_config:
            if name not in self.MCP_SERVERS:
                continue
                
            info = self.MCP_SERVERS[name]
            server_config = {
                "capabilities": info.get("capabilities", [])
            }
            
            if info["type"] == "npm":
                server_config["command"] = "npx"
                server_config["args"] = ["-y", info["package"]]
            else:
                entry_point = self.servers_dir / name / info.get("entry", "index.js")
                server_config["command"] = "node"
                server_config["args"] = [str(entry_point)]
            
            if "env" in info:
                server_config["env"] = info["env"]
            else:
                server_config["env"] = {}
            
            config["mcpServers"][name] = server_config
        
        config_file = self.config_dir / "mcp-config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.print_status(Status.SUCCESS, f"Configuration updated at {config_file}")
    
    def check_server_health(self, name: str) -> bool:
        """Check health of a single MCP server"""
        install_dir = self.servers_dir / name
        
        if not install_dir.exists():
            self.print_status(Status.ERROR, f"{name} not installed")
            return False
        
        server_info = self.MCP_SERVERS.get(name, {})
        
        # Check for entry point
        if server_info.get("type") == "github":
            entry_file = install_dir / server_info.get("entry", "index.js")
            if entry_file.exists():
                # Quick syntax check
                success, _ = self.run_command(["node", "-c", str(entry_file)])
                if success:
                    self.print_status(Status.SUCCESS, f"{name} is healthy")
                    return True
                else:
                    self.print_status(Status.WARNING, f"{name} has syntax errors")
                    return False
        
        # For npm packages, check if package.json exists
        if (install_dir / "package.json").exists():
            self.print_status(Status.SUCCESS, f"{name} appears configured")
            return True
        
        self.print_status(Status.ERROR, f"{name} missing configuration")
        return False
    
    def health_check_all(self):
        """Run health check on all servers"""
        self.print_status(Status.HEADER, "MCP Server Health Check")
        
        healthy = 0
        total = len(self.MCP_SERVERS)
        
        for name in self.MCP_SERVERS:
            if self.check_server_health(name):
                healthy += 1
        
        print()
        self.print_status(Status.INFO, f"Health Summary: {healthy}/{total} servers healthy")
        
        # Check configuration
        config_file = self.config_dir / "mcp-config.json"
        if config_file.exists():
            self.print_status(Status.SUCCESS, "Configuration file exists")
        else:
            self.print_status(Status.WARNING, "Configuration file missing - run install to create")
    
    def update_all(self):
        """Update all installed MCP servers"""
        self.print_status(Status.HEADER, "Updating MCP Servers")
        
        for name in self.MCP_SERVERS:
            install_dir = self.servers_dir / name
            
            if not install_dir.exists():
                self.print_status(Status.WARNING, f"{name} not installed, skipping...")
                continue
            
            self.print_status(Status.INFO, f"Updating {name}...")
            
            if (install_dir / ".git").exists():
                # Git repository - pull updates
                self.run_command(["git", "pull"], cwd=install_dir)
                if (install_dir / "package.json").exists():
                    self.run_command(["npm", "update"], cwd=install_dir)
                    self.run_command(["npm", "install"], cwd=install_dir)
            elif (install_dir / "package.json").exists():
                # NPM package - update
                self.run_command(["npm", "update"], cwd=install_dir)
            
            self.print_status(Status.SUCCESS, f"{name} updated")
        
        self.update_config()
    
    def repair(self, retry_failed_only: bool = True):
        """Repair broken installations
        
        Args:
            retry_failed_only: If True, only retry servers that aren't healthy
        """
        self.print_status(Status.HEADER, "Repairing MCP Installations")
        
        servers_to_repair = []
        healthy_servers = []
        
        # Check which servers need repair
        for name in self.MCP_SERVERS:
            if not self.check_server_health(name):
                servers_to_repair.append(name)
            else:
                healthy_servers.append(name)
        
        if not servers_to_repair:
            self.print_status(Status.SUCCESS, "All servers are healthy - no repair needed")
            return
        
        self.print_status(Status.INFO, f"Found {len(servers_to_repair)} servers needing repair")
        
        # Repair broken servers
        repaired = []
        still_broken = []
        
        for name in servers_to_repair:
            self.print_status(Status.INFO, f"Repairing {name}...")
            if self.install_server(name, skip_on_fail=True):
                repaired.append(name)
            else:
                still_broken.append(name)
        
        # Print repair summary
        print("\n" + "="*50)
        self.print_status(Status.HEADER, "Repair Summary")
        print(f"Servers needing repair: {len(servers_to_repair)}")
        print(f"Successfully repaired: {len(repaired)}")
        print(f"Still broken: {len(still_broken)}")
        
        if repaired:
            print()
            self.print_status(Status.SUCCESS, "Repaired servers:")
            for server in repaired:
                print(f"  ✓ {server}")
        
        if still_broken:
            print()
            self.print_status(Status.WARNING, "Failed to repair:")
            for server in still_broken:
                print(f"  ✗ {server}")
        
        # Update config with all healthy servers
        all_healthy = healthy_servers + repaired
        if all_healthy:
            self.update_config(only_servers=all_healthy)
        
        if still_broken:
            self.print_status(Status.WARNING, "Some servers could not be repaired")
            self.print_status(Status.INFO, f"Check {self.log_file} for error details")
        else:
            self.print_status(Status.SUCCESS, "Repair complete - all servers healthy!")
    
    def clean_install(self):
        """Remove and reinstall all servers"""
        self.print_status(Status.HEADER, "Clean Installation")
        
        response = input("This will remove all existing MCP servers and reinstall. Continue? (y/N) ")
        if response.lower() != 'y':
            self.print_status(Status.INFO, "Clean installation cancelled")
            return
        
        self.print_status(Status.WARNING, "Removing existing installations...")
        
        if self.servers_dir.exists():
            shutil.rmtree(self.servers_dir)
        
        config_file = self.config_dir / "mcp-config.json"
        if config_file.exists():
            config_file.unlink()
        
        self.setup_directories()
        self.install_all()
    
    def show_status(self):
        """Show current status of MCP servers"""
        self.print_status(Status.HEADER, "MCP Server Status")
        
        print(f"\nConfiguration Directory: {self.config_dir}")
        print(f"Servers Directory: {self.servers_dir}")
        print(f"Log File: {self.log_file}")
        print("\nRegistered Servers:")
        
        for name in self.MCP_SERVERS:
            install_dir = self.servers_dir / name
            if install_dir.exists():
                print(f"  • {name}: Installed")
            else:
                print(f"  • {name}: Not installed")
        
        # Check if config exists
        config_file = self.config_dir / "mcp-config.json"
        if config_file.exists():
            print(f"\nConfiguration: {Color.GREEN}Present{Color.NC}")
        else:
            print(f"\nConfiguration: {Color.RED}Missing{Color.NC}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MCP Server Manager - Manage Model Context Protocol servers for Claude Code"
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        choices=['install', 'update', 'check', 'health', 'repair', 'clean', 'status'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--server', '-s',
        help='Specific server to operate on'
    )
    
    args = parser.parse_args()
    
    # Create manager instance
    manager = MCPServerManager(verbose=args.verbose)
    
    # Check prerequisites
    if not manager.check_prerequisites():
        manager.print_status(Status.ERROR, "Prerequisites not met. Please install missing dependencies.")
        sys.exit(1)
    
    # Handle actions
    if args.action == 'install':
        if args.server:
            manager.install_server(args.server)
        else:
            manager.install_all()
    elif args.action == 'update':
        manager.update_all()
    elif args.action in ['check', 'health']:
        manager.health_check_all()
    elif args.action == 'repair':
        manager.repair()
    elif args.action == 'clean':
        manager.clean_install()
    elif args.action == 'status':
        manager.show_status()
    else:
        # Interactive mode
        while True:
            print("\n" + "="*50)
            print("MCP Server Manager")
            print("="*50)
            print("1) Install all MCP servers")
            print("2) Update all MCP servers")
            print("3) Health check")
            print("4) Repair broken installations")
            print("5) Clean install (remove and reinstall)")
            print("6) Show status")
            print("7) Exit")
            
            choice = input("\nEnter choice [1-7]: ")
            
            if choice == '1':
                manager.install_all()
            elif choice == '2':
                manager.update_all()
            elif choice == '3':
                manager.health_check_all()
            elif choice == '4':
                manager.repair()
            elif choice == '5':
                manager.clean_install()
            elif choice == '6':
                manager.show_status()
            elif choice == '7':
                print("Exiting...")
                break
            else:
                print("Invalid choice")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()