#!/usr/bin/env python3
"""
Create distributable package for BeastClipper Ultimate
This script packages all files into a ready-to-share ZIP file
"""

import os
import shutil
import zipfile
from datetime import datetime
import json

class PackageCreator:
    def __init__(self):
        self.package_name = f"BeastClipper_Ultimate_v3.0_{datetime.now().strftime('%Y%m%d')}"
        self.files_to_include = [
            # Core files
            "main.py",
            "config.py",
            "stream.py",
            "analysis.py",
            "export.py",
            "detection.py",
            "credentials.py",
            
            # Ultimate detection system
            "ultimate_viral_system.py",
            "twitch_chat_detector.py",
            "social_proof_analyzer.py",
            "context_timing_system.py",
            "analytics_learning_system.py",
            "enhanced_detection.py",
            
            # Analytics UI
            "analytics_dashboard_ui.py",
            
            # Installer files
            "install_beastclipper.py",
            "INSTALL_WINDOWS.bat",
            "requirements.txt",
            "README.md",
            
            # Package info
            "package_info.json"
        ]
        
        self.directories_to_create = [
            "temp",
            "temp/buffer",
            "temp/logs",
            "clips",
            "profiles",
            "exports"
        ]
    
    def create_package_info(self):
        """Create package information file"""
        info = {
            "name": "BeastClipper Ultimate",
            "version": "3.0",
            "build_date": datetime.now().isoformat(),
            "author": "BeastClipper Team",
            "description": "Ultimate viral clip detection tool for Twitch streamers",
            "python_version": "3.7+",
            "features": [
                "Real-time Twitch stream buffering",
                "AI-powered viral moment detection",
                "Multi-signal analysis (chat, video, social)",
                "Perfect timing optimization",
                "Multi-platform export (YouTube, Twitch, TikTok)",
                "VOD analysis support",
                "Learning system that adapts to your content",
                "Comprehensive analytics dashboard"
            ],
            "requirements": {
                "os": ["Windows 10+", "macOS 10.14+", "Ubuntu 20.04+"],
                "ram": "4GB minimum, 8GB recommended",
                "storage": "10GB free space",
                "internet": "Stable broadband connection"
            }
        }
        
        with open("package_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print("✅ Created package_info.json")
    
    def verify_files(self):
        """Verify all required files exist"""
        print("\n📋 Verifying files...")
        missing_files = []
        
        for file in self.files_to_include:
            if not os.path.exists(file):
                missing_files.append(file)
                print(f"❌ Missing: {file}")
            else:
                print(f"✅ Found: {file}")
        
        if missing_files:
            print(f"\n⚠️  Warning: {len(missing_files)} files are missing!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        return True
    
    def create_package(self):
        """Create the distributable package"""
        print(f"\n📦 Creating package: {self.package_name}.zip")
        
        # Create temporary package directory
        package_dir = self.package_name
        if os.path.exists(package_dir):
            shutil.rmtree(package_dir)
        os.makedirs(package_dir)
        
        # Copy files
        print("\n📂 Copying files...")
        for file in self.files_to_include:
            if os.path.exists(file):
                dst = os.path.join(package_dir, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(file, dst)
                print(f"  - Copied: {file}")
        
        # Create empty directories
        print("\n📁 Creating directories...")
        for directory in self.directories_to_create:
            dir_path = os.path.join(package_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            # Add .gitkeep file to preserve empty directories
            with open(os.path.join(dir_path, ".gitkeep"), "w") as f:
                f.write("")
        
        # Create default config file
        print("\n⚙️  Creating default config...")
        default_config = {
            "output_directory": "clips",
            "buffer_duration": 300,
            "format": "mp4",
            "resolution": "1080p",
            "viral_detection": {
                "enabled": True,
                "sensitivity": 0.7,
                "mode": "balanced",
                "auto_clip_high_confidence": False
            }
        }
        
        config_path = os.path.join(package_dir, ".beastclipper_config.json")
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        # Create ZIP file
        print(f"\n🗜️  Creating ZIP archive...")
        zip_filename = f"{self.package_name}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(package_dir))
                    zipf.write(file_path, arcname)
                    print(f"  - Added: {arcname}")
        
        # Clean up temporary directory
        shutil.rmtree(package_dir)
        
        # Get file size
        file_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
        
        print("\n" + "="*60)
        print("✅ PACKAGE CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"📦 Package: {zip_filename}")
        print(f"📏 Size: {file_size:.2f} MB")
        print("\n📤 This package is ready to share!")
        print("\nRecipients should:")
        print("1. Extract the ZIP file")
        print("2. Run INSTALL_WINDOWS.bat (Windows) or python install_beastclipper.py")
        print("3. Follow the installation prompts")
        print("="*60)
        
        return True
    
    def run(self):
        """Run the package creation process"""
        print("="*60)
        print("🔥 BEASTCLIPPER ULTIMATE PACKAGE CREATOR")
        print("="*60)
        
        # Create package info
        self.create_package_info()
        
        # Verify files
        if not self.verify_files():
            print("\n❌ Package creation cancelled.")
            return False
        
        # Create package
        if self.create_package():
            return True
        
        return False


def main():
    """Main entry point"""
    creator = PackageCreator()
    
    print("\nThis will create a distributable package of BeastClipper Ultimate.")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        success = creator.run()
        if not success:
            print("\n❌ Package creation failed.")
    else:
        print("\n❌ Package creation cancelled.")


if __name__ == "__main__":
    main()
