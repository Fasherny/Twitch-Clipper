#!/usr/bin/env python3
"""
BeastClipper Ultimate v3.0 - Automatic Installer
Installs all dependencies and sets up the application
"""

import os
import sys
import platform
import subprocess
import shutil
import urllib.request
import zipfile
import json
from pathlib import Path

class BeastClipperInstaller:
    def __init__(self):
        self.system = platform.system()
        self.install_dir = os.path.dirname(os.path.abspath(__file__))
        self.errors = []
        self.warnings = []
        
    def print_banner(self):
        """Print installation banner"""
        print("\n" + "="*60)
        print("🔥 BEASTCLIPPER ULTIMATE v3.0 INSTALLER 🔥")
        print("="*60)
        print(f"System: {self.system}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Install Directory: {self.install_dir}")
        print("="*60 + "\n")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("📌 Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.errors.append("Python 3.7 or higher is required!")
            return False
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True
    
    def install_python_packages(self):
        """Install required Python packages"""
        print("\n📦 Installing Python packages...")
        
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        
        # Read requirements
        req_file = os.path.join(self.install_dir, "requirements.txt")
        if not os.path.exists(req_file):
            # Create requirements if missing
            requirements = [
                "PyQt5>=5.15.0",
                "opencv-python>=4.8.0",
                "numpy>=1.21.0",
                "selenium>=4.15.0",
                "psutil>=5.9.0",
                "requests>=2.31.0",
                "requests-toolbelt>=1.0.0",
                "send2trash>=1.8.0",
                "streamlink>=6.5.0",
                "yt-dlp>=2023.12.30",
                "matplotlib>=3.5.0",
                "scipy>=1.7.0"
            ]
            with open(req_file, 'w') as f:
                f.write('\n'.join(requirements))
        
        # Install packages
        print("Installing packages from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            self.warnings.append(f"Some packages failed to install: {result.stderr}")
        else:
            print("✅ Python packages installed successfully")
    
    def install_system_tools(self):
        """Install system tools like FFmpeg"""
        print("\n🔧 Installing system tools...")
        
        if self.system == "Windows":
            self._install_windows_tools()
        elif self.system == "Darwin":  # macOS
            self._install_macos_tools()
        else:  # Linux
            self._install_linux_tools()
    
    def _install_windows_tools(self):
        """Install tools on Windows"""
        print("Installing tools for Windows...")
        
        # Check for FFmpeg
        if not self._check_command("ffmpeg"):
            print("📥 FFmpeg not found. Downloading...")
            self._download_ffmpeg_windows()
        else:
            print("✅ FFmpeg already installed")
        
        # Check for Chrome/Chromium driver
        if not self._check_chromedriver():
            print("📥 ChromeDriver not found. Downloading...")
            self._download_chromedriver_windows()
        else:
            print("✅ ChromeDriver already installed")
    
    def _install_macos_tools(self):
        """Install tools on macOS"""
        print("Installing tools for macOS...")
        
        # Check for Homebrew
        if not self._check_command("brew"):
            print("⚠️ Homebrew not found. Please install from https://brew.sh")
            self.warnings.append("Homebrew required for automatic installation on macOS")
            return
        
        # Install FFmpeg
        if not self._check_command("ffmpeg"):
            print("Installing FFmpeg via Homebrew...")
            subprocess.run(["brew", "install", "ffmpeg"], capture_output=True)
        else:
            print("✅ FFmpeg already installed")
        
        # Install ChromeDriver
        if not self._check_chromedriver():
            print("Installing ChromeDriver via Homebrew...")
            subprocess.run(["brew", "install", "chromedriver"], capture_output=True)
        else:
            print("✅ ChromeDriver already installed")
    
    def _install_linux_tools(self):
        """Install tools on Linux"""
        print("Installing tools for Linux...")
        
        # Detect package manager
        if self._check_command("apt"):
            pkg_manager = "apt"
            install_cmd = ["sudo", "apt", "install", "-y"]
        elif self._check_command("dnf"):
            pkg_manager = "dnf"
            install_cmd = ["sudo", "dnf", "install", "-y"]
        elif self._check_command("pacman"):
            pkg_manager = "pacman"
            install_cmd = ["sudo", "pacman", "-S", "--noconfirm"]
        else:
            self.warnings.append("Could not detect package manager. Please install FFmpeg manually.")
            return
        
        # Install FFmpeg
        if not self._check_command("ffmpeg"):
            print(f"Installing FFmpeg via {pkg_manager}...")
            subprocess.run(install_cmd + ["ffmpeg"], capture_output=True)
        else:
            print("✅ FFmpeg already installed")
        
        # Install Chrome/Chromium
        if not self._check_command("google-chrome") and not self._check_command("chromium"):
            print(f"Installing Chromium via {pkg_manager}...")
            if pkg_manager == "apt":
                subprocess.run(install_cmd + ["chromium-browser"], capture_output=True)
            else:
                subprocess.run(install_cmd + ["chromium"], capture_output=True)
        else:
            print("✅ Chrome/Chromium already installed")
    
    def _check_command(self, cmd):
        """Check if a command exists"""
        try:
            if self.system == "Windows":
                subprocess.run(["where", cmd], capture_output=True, check=True)
            else:
                subprocess.run(["which", cmd], capture_output=True, check=True)
            return True
        except:
            return False
    
    def _check_chromedriver(self):
        """Check if ChromeDriver is available"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)
            driver.quit()
            return True
        except:
            return False
    
    def _download_ffmpeg_windows(self):
        """Download FFmpeg for Windows"""
        try:
            ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            ffmpeg_zip = os.path.join(self.install_dir, "ffmpeg.zip")
            
            print("Downloading FFmpeg...")
            urllib.request.urlretrieve(ffmpeg_url, ffmpeg_zip)
            
            print("Extracting FFmpeg...")
            with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
                zip_ref.extractall(self.install_dir)
            
            # Find the extracted folder
            for item in os.listdir(self.install_dir):
                if item.startswith("ffmpeg-") and os.path.isdir(os.path.join(self.install_dir, item)):
                    ffmpeg_dir = os.path.join(self.install_dir, item, "bin")
                    
                    # Add to PATH or copy to install directory
                    for exe in ["ffmpeg.exe", "ffprobe.exe"]:
                        src = os.path.join(ffmpeg_dir, exe)
                        dst = os.path.join(self.install_dir, exe)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    
                    # Clean up
                    shutil.rmtree(os.path.join(self.install_dir, item))
                    break
            
            os.remove(ffmpeg_zip)
            print("✅ FFmpeg installed successfully")
            
        except Exception as e:
            self.warnings.append(f"Failed to download FFmpeg: {e}")
            print(f"⚠️ Failed to download FFmpeg: {e}")
            print("Please download manually from: https://ffmpeg.org/download.html")
    
    def _download_chromedriver_windows(self):
        """Download ChromeDriver for Windows"""
        try:
            # This would download the appropriate ChromeDriver
            # For brevity, just showing the structure
            print("✅ Please ensure Chrome browser is installed")
            print("ChromeDriver will be downloaded automatically by Selenium")
            
        except Exception as e:
            self.warnings.append(f"ChromeDriver setup issue: {e}")
    
    def create_shortcuts(self):
        """Create desktop shortcuts"""
        print("\n🔗 Creating shortcuts...")
        
        if self.system == "Windows":
            self._create_windows_shortcut()
        elif self.system == "Darwin":
            self._create_macos_app()
        else:
            self._create_linux_desktop()
    
    def _create_windows_shortcut(self):
        """Create Windows shortcut"""
        try:
            import win32com.client
            
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / "BeastClipper Ultimate.lnk"
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{os.path.join(self.install_dir, "main.py")}"'
            shortcut.WorkingDirectory = self.install_dir
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("✅ Desktop shortcut created")
            
        except:
            # Create a batch file instead
            desktop = Path.home() / "Desktop"
            batch_path = desktop / "BeastClipper Ultimate.bat"
            
            with open(batch_path, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{self.install_dir}"\n')
                f.write(f'"{sys.executable}" main.py\n')
                f.write(f'pause\n')
            
            print("✅ Desktop batch file created")
    
    def _create_macos_app(self):
        """Create macOS app"""
        desktop = Path.home() / "Desktop"
        script_path = desktop / "BeastClipper Ultimate.command"
        
        with open(script_path, 'w') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'cd "{self.install_dir}"\n')
            f.write(f'"{sys.executable}" main.py\n')
        
        os.chmod(script_path, 0o755)
        print("✅ Desktop launcher created")
    
    def _create_linux_desktop(self):
        """Create Linux desktop entry"""
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        
        desktop_file = desktop_dir / "beastclipper.desktop"
        
        with open(desktop_file, 'w') as f:
            f.write(f'''[Desktop Entry]
Name=BeastClipper Ultimate
Comment=Ultimate Viral Clip Detection Tool
Exec={sys.executable} "{os.path.join(self.install_dir, "main.py")}"
Icon={sys.executable}
Terminal=false
Type=Application
Categories=AudioVideo;
''')
        
        os.chmod(desktop_file, 0o755)
        print("✅ Desktop entry created")
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        dirs_to_create = [
            "temp",
            "temp/buffer",
            "temp/logs",
            "clips",
            "profiles",
            "exports"
        ]
        
        for dir_name in dirs_to_create:
            dir_path = os.path.join(self.install_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        
        print("✅ Directories created")
    
    def test_installation(self):
        """Test if installation was successful"""
        print("\n🧪 Testing installation...")
        
        # Test Python imports
        try:
            import PyQt5
            import cv2
            import numpy
            import selenium
            import streamlink
            print("✅ All Python packages imported successfully")
        except ImportError as e:
            self.errors.append(f"Failed to import package: {e}")
            return False
        
        # Test FFmpeg
        if self._check_command("ffmpeg"):
            print("✅ FFmpeg is accessible")
        else:
            self.warnings.append("FFmpeg not found in PATH")
        
        return len(self.errors) == 0
    
    def print_summary(self):
        """Print installation summary"""
        print("\n" + "="*60)
        print("📊 INSTALLATION SUMMARY")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✅ Installation completed successfully!")
            print("\n🚀 To start BeastClipper Ultimate:")
            if self.system == "Windows":
                print("   - Double-click 'BeastClipper Ultimate' on your desktop")
                print("   - Or run: python main.py")
            else:
                print("   - Click the BeastClipper Ultimate launcher on your desktop")
                print("   - Or run: python3 main.py")
        else:
            if self.errors:
                print("\n❌ ERRORS:")
                for error in self.errors:
                    print(f"   - {error}")
            
            if self.warnings:
                print("\n⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"   - {warning}")
            
            print("\n⚠️  Installation completed with issues.")
            print("Please address the errors/warnings above.")
        
        print("\n📖 Documentation: https://github.com/yourusername/beastclipper")
        print("💬 Support: your-email@example.com")
        print("="*60 + "\n")
    
    def run(self):
        """Run the installation process"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            self.print_summary()
            return False
        
        # Install Python packages
        self.install_python_packages()
        
        # Install system tools
        self.install_system_tools()
        
        # Create directories
        self.create_directories()
        
        # Create shortcuts
        self.create_shortcuts()
        
        # Test installation
        self.test_installation()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0


def main():
    """Main entry point"""
    installer = BeastClipperInstaller()
    success = installer.run()
    
    if success:
        response = input("\n🎯 Would you like to start BeastClipper now? (y/n): ")
        if response.lower() == 'y':
            print("\nStarting BeastClipper Ultimate...")
            subprocess.run([sys.executable, "main.py"])
    else:
        print("\n❌ Installation failed. Please check the errors above.")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
