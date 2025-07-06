#!/usr/bin/env python3
"""
Lean-Lite Project Structure Verification Script

This script validates the complete Lean-Lite project setup to ensure
all required components are present and properly configured.
"""

import os
import sys
import importlib
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ProjectVerifier:
    """Verifies the Lean-Lite project structure and configuration."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.passed = 0
        self.total = 0
        
    def log_pass(self, message: str):
        """Log a successful verification."""
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
        self.passed += 1
        self.total += 1
    
    def log_fail(self, message: str):
        """Log a failed verification."""
        print(f"{Colors.RED}✗{Colors.END} {message}")
        self.errors.append(message)
        self.total += 1
    
    def log_warning(self, message: str):
        """Log a warning."""
        print(f"{Colors.YELLOW}⚠{Colors.END} {message}")
        self.warnings.append(message)
    
    def log_info(self, message: str):
        """Log an informational message."""
        print(f"{Colors.BLUE}ℹ{Colors.END} {message}")
    
    def verify_directories(self) -> bool:
        """Verify that all required directories exist."""
        print(f"\n{Colors.BOLD}Verifying Directory Structure:{Colors.END}")
        
        required_dirs = [
            "src",
            "src/lean_lite",
            "src/lean_lite/algorithm",
            "src/lean_lite/data",
            "src/lean_lite/brokers",
            "src/lean_lite/engine",
            "src/lean_lite/indicators",
            "strategies",
            "docker",
            "tests"
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_pass(f"Directory exists: {dir_path}")
            else:
                self.log_fail(f"Directory missing: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def verify_init_files(self) -> bool:
        """Verify that all required __init__.py files exist."""
        print(f"\n{Colors.BOLD}Verifying __init__.py Files:{Colors.END}")
        
        required_init_files = [
            "src/__init__.py",
            "src/lean_lite/__init__.py",
            "src/lean_lite/algorithm/__init__.py",
            "src/lean_lite/data/__init__.py",
            "src/lean_lite/brokers/__init__.py",
            "src/lean_lite/engine/__init__.py",
            "src/lean_lite/indicators/__init__.py",
            "strategies/__init__.py",
            "tests/__init__.py"
        ]
        
        all_exist = True
        for init_file in required_init_files:
            full_path = self.project_root / init_file
            if full_path.exists() and full_path.is_file():
                self.log_pass(f"__init__.py exists: {init_file}")
            else:
                self.log_fail(f"__init__.py missing: {init_file}")
                all_exist = False
        
        return all_exist
    
    def verify_requirements_txt(self) -> bool:
        """Verify requirements.txt contains all specified dependencies."""
        print(f"\n{Colors.BOLD}Verifying requirements.txt:{Colors.END}")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.log_fail("requirements.txt file missing")
            return False
        
        required_deps = [
            "alpaca-py",
            "websocket-client",
            "numpy",
            "pandas",
            "asyncio"
        ]
        
        try:
            with open(requirements_file, 'r') as f:
                content = f.read()
            
            missing_deps = []
            for dep in required_deps:
                if dep in content:
                    self.log_pass(f"Dependency found: {dep}")
                else:
                    self.log_fail(f"Dependency missing: {dep}")
                    missing_deps.append(dep)
            
            return len(missing_deps) == 0
            
        except Exception as e:
            self.log_fail(f"Error reading requirements.txt: {e}")
            return False
    
    def verify_pyproject_toml(self) -> bool:
        """Verify pyproject.toml exists and has correct structure."""
        print(f"\n{Colors.BOLD}Verifying pyproject.toml:{Colors.END}")
        
        pyproject_file = self.project_root / "pyproject.toml"
        if not pyproject_file.exists():
            self.log_fail("pyproject.toml file missing")
            return False
        
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                "[tool.poetry]",
                "[tool.poetry.dependencies]",
                "name = \"lean-lite\"",
                "alpaca-py"
            ]
            
            all_found = True
            for section in required_sections:
                if section in content:
                    self.log_pass(f"Section found: {section}")
                else:
                    self.log_fail(f"Section missing: {section}")
                    all_found = False
            
            return all_found
            
        except Exception as e:
            self.log_fail(f"Error reading pyproject.toml: {e}")
            return False
    
    def verify_main_py(self) -> bool:
        """Verify main.py exists and has proper structure."""
        print(f"\n{Colors.BOLD}Verifying main.py:{Colors.END}")
        
        main_file = self.project_root / "src" / "main.py"
        if not main_file.exists():
            self.log_fail("main.py file missing")
            return False
        
        try:
            with open(main_file, 'r') as f:
                content = f.read()
            
            # Check for required elements
            required_elements = [
                "#!/usr/bin/env python3",
                "def main():",
                "if __name__ == \"__main__\":",
                "from lean_lite.engine import LeanEngine",
                "from lean_lite.config import Config"
            ]
            
            all_found = True
            for element in required_elements:
                if element in content:
                    self.log_pass(f"Element found: {element}")
                else:
                    self.log_fail(f"Element missing: {element}")
                    all_found = False
            
            # Check for proper imports
            try:
                tree = ast.parse(content)
                imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
                if imports:
                    self.log_pass("Python syntax is valid")
                else:
                    self.log_warning("No imports found in main.py")
            except SyntaxError as e:
                self.log_fail(f"Python syntax error in main.py: {e}")
                all_found = False
            
            return all_found
            
        except Exception as e:
            self.log_fail(f"Error reading main.py: {e}")
            return False
    
    def verify_readme_md(self) -> bool:
        """Verify README.md exists and contains project description."""
        print(f"\n{Colors.BOLD}Verifying README.md:{Colors.END}")
        
        readme_file = self.project_root / "README.md"
        if not readme_file.exists():
            self.log_fail("README.md file missing")
            return False
        
        try:
            with open(readme_file, 'r') as f:
                content = f.read()
            
            # Check for required content
            required_content = [
                "# Lean-Lite",
                "A lightweight, containerized QuantConnect LEAN runtime",
                "## Overview",
                "## Features",
                "## Quick Start"
            ]
            
            all_found = True
            for content_item in required_content:
                if content_item in content:
                    self.log_pass(f"Content found: {content_item}")
                else:
                    self.log_fail(f"Content missing: {content_item}")
                    all_found = False
            
            return all_found
            
        except Exception as e:
            self.log_fail(f"Error reading README.md: {e}")
            return False
    
    def verify_python_imports(self) -> bool:
        """Verify that the project can be imported as a Python package."""
        print(f"\n{Colors.BOLD}Verifying Python Package Imports:{Colors.END}")
        
        # Add src to Python path
        src_path = self.project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Test imports - separate core imports from optional ones
        core_imports = [
            ("lean_lite", "Main package"),
            ("lean_lite.config", "Config module"),
            ("lean_lite.engine", "Engine module"),
            ("lean_lite.algorithm", "Algorithm module"),
            ("lean_lite.brokers", "Brokers module"),
            ("lean_lite.data", "Data module"),
            ("lean_lite.indicators", "Indicators module"),
        ]
        
        optional_imports = [
            ("lean_lite.algorithm.qc_algorithm", "QCAlgorithm module"),
        ]
        
        all_successful = True
        
        # Test core imports
        for module_name, description in core_imports:
            try:
                module = importlib.import_module(module_name)
                self.log_pass(f"Import successful: {description}")
            except ImportError as e:
                if "alpaca" in str(e):
                    self.log_warning(f"Import failed (missing alpaca-py): {description} - {e}")
                    self.log_info("Install alpaca-py with: pip install alpaca-py>=0.35.0")
                else:
                    self.log_fail(f"Import failed: {description} - {e}")
                    all_successful = False
            except Exception as e:
                self.log_fail(f"Unexpected error importing {description}: {e}")
                all_successful = False
        
        # Test optional imports
        for module_name, description in optional_imports:
            try:
                module = importlib.import_module(module_name)
                self.log_pass(f"Import successful: {description}")
            except ImportError as e:
                if "alpaca" in str(e):
                    self.log_warning(f"Optional import failed (missing alpaca-py): {description} - {e}")
                else:
                    self.log_warning(f"Optional import failed: {description} - {e}")
            except Exception as e:
                self.log_warning(f"Unexpected error importing {description}: {e}")
        
        return all_successful
    
    def verify_python_path(self) -> bool:
        """Test that Python path is correctly configured."""
        print(f"\n{Colors.BOLD}Verifying Python Path Configuration:{Colors.END}")
        
        # Test that we can access the project root
        if str(self.project_root) in sys.path or str(self.project_root / "src") in sys.path:
            self.log_pass("Project path is in sys.path")
            return True
        else:
            self.log_warning("Project path not in sys.path (this is normal for verification script)")
            return True
    
    def verify_docker_files(self) -> bool:
        """Verify Docker configuration files exist."""
        print(f"\n{Colors.BOLD}Verifying Docker Configuration:{Colors.END}")
        
        docker_files = [
            "docker/Dockerfile",
            "docker/docker-compose.yml"
        ]
        
        all_exist = True
        for docker_file in docker_files:
            full_path = self.project_root / docker_file
            if full_path.exists() and full_path.is_file():
                self.log_pass(f"Docker file exists: {docker_file}")
            else:
                self.log_fail(f"Docker file missing: {docker_file}")
                all_exist = False
        
        return all_exist
    
    def verify_gitignore(self) -> bool:
        """Verify .gitignore file exists."""
        print(f"\n{Colors.BOLD}Verifying .gitignore:{Colors.END}")
        
        gitignore_file = self.project_root / ".gitignore"
        if gitignore_file.exists() and gitignore_file.is_file():
            self.log_pass(".gitignore file exists")
            return True
        else:
            self.log_fail(".gitignore file missing")
            return False
    
    def verify_test_files(self) -> bool:
        """Verify that test files exist."""
        print(f"\n{Colors.BOLD}Verifying Test Files:{Colors.END}")
        
        test_files = [
            "tests/test_config.py",
            "tests/test_alpaca_broker.py",
            "tests/test_qc_algorithm.py"
        ]
        
        all_exist = True
        for test_file in test_files:
            full_path = self.project_root / test_file
            if full_path.exists() and full_path.is_file():
                self.log_pass(f"Test file exists: {test_file}")
            else:
                self.log_fail(f"Test file missing: {test_file}")
                all_exist = False
        
        return all_exist
    
    def verify_strategy_files(self) -> bool:
        """Verify that strategy files exist."""
        print(f"\n{Colors.BOLD}Verifying Strategy Files:{Colors.END}")
        
        strategy_files = [
            "strategies/example_strategy.py",
            "strategies/qc_example_strategy.py"
        ]
        
        all_exist = True
        for strategy_file in strategy_files:
            full_path = self.project_root / strategy_file
            if full_path.exists() and full_path.is_file():
                self.log_pass(f"Strategy file exists: {strategy_file}")
            else:
                self.log_fail(f"Strategy file missing: {strategy_file}")
                all_exist = False
        
        return all_exist
    
    def run_all_verifications(self) -> bool:
        """Run all verification checks."""
        print(f"{Colors.BOLD}Lean-Lite Project Structure Verification{Colors.END}")
        print("=" * 50)
        
        verifications = [
            ("Directory Structure", self.verify_directories),
            ("__init__.py Files", self.verify_init_files),
            ("requirements.txt", self.verify_requirements_txt),
            ("pyproject.toml", self.verify_pyproject_toml),
            ("main.py", self.verify_main_py),
            ("README.md", self.verify_readme_md),
            ("Python Package Imports", self.verify_python_imports),
            ("Python Path Configuration", self.verify_python_path),
            ("Docker Configuration", self.verify_docker_files),
            (".gitignore", self.verify_gitignore),
            ("Test Files", self.verify_test_files),
            ("Strategy Files", self.verify_strategy_files),
        ]
        
        all_passed = True
        for name, verification_func in verifications:
            try:
                if not verification_func():
                    all_passed = False
            except Exception as e:
                self.log_fail(f"Error during {name} verification: {e}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print verification summary."""
        print(f"\n{Colors.BOLD}Verification Summary:{Colors.END}")
        print("=" * 30)
        
        if self.errors:
            print(f"{Colors.RED}Errors ({len(self.errors)}):{Colors.END}")
            for error in self.errors:
                print(f"  {Colors.RED}• {error}{Colors.END}")
        
        if self.warnings:
            print(f"{Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings:
                print(f"  {Colors.YELLOW}• {warning}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  Passed: {Colors.GREEN}{self.passed}{Colors.END}")
        print(f"  Failed: {Colors.RED}{len(self.errors)}{Colors.END}")
        print(f"  Warnings: {Colors.YELLOW}{len(self.warnings)}{Colors.END}")
        print(f"  Total: {self.total}")
        
        if self.total > 0:
            success_rate = (self.passed / self.total) * 100
            print(f"  Success Rate: {Colors.GREEN}{success_rate:.1f}%{Colors.END}")
        
        # Provide guidance for warnings
        if self.warnings and any("alpaca" in warning for warning in self.warnings):
            print(f"\n{Colors.BLUE}Next Steps:{Colors.END}")
            print("  To resolve alpaca-py import warnings, install the package:")
            print("  pip install alpaca-py>=0.35.0")
        
        if self.errors:
            print(f"\n{Colors.RED}❌ Verification Failed{Colors.END}")
            return False
        else:
            print(f"\n{Colors.GREEN}✅ Verification Passed{Colors.END}")
            if self.warnings:
                print(f"{Colors.YELLOW}⚠️  Some warnings were found (see above){Colors.END}")
            return True


def main():
    """Main verification function."""
    verifier = ProjectVerifier()
    
    try:
        success = verifier.run_all_verifications()
        verifier.print_summary()
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error during verification: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main() 