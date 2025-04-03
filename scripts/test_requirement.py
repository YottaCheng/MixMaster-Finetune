#!/usr/bin/env python3
"""
Mix Master Dependency Checker
This script verifies that all required dependencies for Mix Master are properly installed.
"""

import importlib
import sys
import subprocess
import pkg_resources
from typing import Dict, List, Tuple

# Define required packages with their minimum versions
REQUIRED_PACKAGES = {
    "python": "3.9",
    "streamlit": "0.0.0",  # Any version is fine for basic check
    "dashscope": "0.0.0",
    "pandas": "0.0.0",
    "torch": "0.0.0",
}

def check_python_version() -> Tuple[bool, str]:
    """Check if the current Python version meets the requirements."""
    current_version = '.'.join(map(str, sys.version_info[:2]))
    required_version = REQUIRED_PACKAGES["python"]
    
    if sys.version_info >= tuple(map(int, required_version.split('.'))):
        return True, f"Python {current_version} (✓ meets required {required_version}+)"
    else:
        return False, f"Python {current_version} (✗ required {required_version}+)"

def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed and get its version."""
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            version = package.__version__
        elif hasattr(package, 'version'):
            version = package.version
        else:
            try:
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "unknown version"
        
        return True, f"{package_name} {version} (✓)"
    except ImportError:
        return False, f"{package_name} (✗ not installed)"

def check_internet_connection() -> Tuple[bool, str]:
    """Test internet connectivity by pinging a reliable server."""
    try:
        # Try to reach Google's DNS server
        subprocess.check_call(
            ["ping", "-c", "1", "8.8.8.8"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        return True, "Internet connection (✓)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Alternative check if ping is not available
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=1)
            return True, "Internet connection (✓)"
        except:
            return False, "Internet connection (✗ not available)"

def check_gpu_availability() -> Tuple[bool, str]:
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA available (✓ {torch.cuda.get_device_name(0)})"
        else:
            return False, "CUDA (✗ not available)"
    except:
        return False, "CUDA check failed (torch not installed or other error)"

def main():
    """Run all checks and display results."""
    print("\n=== Mix Master Dependency Checker ===\n")
    
    all_checks_passed = True
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    all_checks_passed = all_checks_passed and python_ok
    print(python_msg)
    
    # Check for required packages
    print("\nChecking required packages:")
    for package in ["streamlit", "dashscope", "pandas", "torch"]:
        pkg_ok, pkg_msg = check_package_installed(package)
        all_checks_passed = all_checks_passed and pkg_ok
        print(f"  - {pkg_msg}")
    
    # Check internet connection
    print("\nChecking system requirements:")
    net_ok, net_msg = check_internet_connection()
    all_checks_passed = all_checks_passed and net_ok
    print(f"  - {net_msg}")
    
    # Check GPU/CUDA availability
    gpu_ok, gpu_msg = check_gpu_availability()
    # Not required but nice to have
    print(f"  - {gpu_msg}")
    
    # Print summary
    print("\n=== Summary ===")
    if all_checks_passed:
        print("✅ All required dependencies are installed!")
        print("You can run Mix Master with the following command:")
        print("    streamlit run /path/to/MixMaster-Finetune/scripts/web/app_streamlit.py")
    else:
        print("❌ Some required dependencies are missing.")
        print("Please install missing dependencies using:")
        print("    pip install -r requirements.txt")
        print("Or follow the installation instructions in the user guide.")
    
    print("\n=======================================\n")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())