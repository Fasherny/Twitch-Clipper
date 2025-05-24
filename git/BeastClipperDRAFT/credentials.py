# credentials.py - Add this file to your project
"""
Simple credentials manager for BeastClipper
Handles basic authentication for all social platforms
"""

import os
import json
import base64
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger("BeastClipper")

class CredentialsManager:
    """Manages user credentials for various platforms."""
    
    def __init__(self, config_manager):
        """Initialize credentials manager."""
        self.config_manager = config_manager
        self.credentials = {}
        self.load_credentials()
    
    def load_credentials(self):
        """Load stored credentials from config."""
        # Load encrypted credentials from config
        encrypted_creds = self.config_manager.get("encrypted_credentials", {})
        
        # Decrypt and store
        for platform, encrypted_data in encrypted_creds.items():
            try:
                if encrypted_data:
                    decrypted = self._decrypt(encrypted_data)
                    self.credentials[platform] = json.loads(decrypted)
            except Exception as e:
                logger.error(f"Error loading credentials for {platform}: {e}")
    
    def save_credentials(self):
        """Save credentials to config securely."""
        # Encrypt all credentials
        encrypted_creds = {}
        for platform, creds in self.credentials.items():
            if creds:
                encrypted = self._encrypt(json.dumps(creds))
                encrypted_creds[platform] = encrypted
        
        # Save to config
        self.config_manager.set("encrypted_credentials", encrypted_creds)
    
    def _encrypt(self, data):
        """Simple encryption for credentials."""
        # This is a very basic obfuscation, not true encryption
        # Just to avoid storing plaintext passwords in config
        return base64.b64encode(data.encode()).decode()
    
    def _decrypt(self, encrypted_data):
        """Decrypt credentials data."""
        # Decode the basic obfuscation
        return base64.b64decode(encrypted_data.encode()).decode()
    
    def set_credentials(self, platform, username, password=None, token=None, url=None):
        """Store credentials for a platform."""
        creds = {"username": username}
        
        if password:
            creds["password"] = password
        
        if token:
            creds["token"] = token
            
        if url:
            creds["url"] = url
        
        self.credentials[platform] = creds
        self.save_credentials()
        
        logger.info(f"Saved credentials for {platform}")
        return True
    
    def get_credentials(self, platform):
        """Get credentials for a platform."""
        return self.credentials.get(platform, {})
    
    def has_credentials(self, platform):
        """Check if credentials exist for a platform."""
        return platform in self.credentials and bool(self.credentials[platform])
    
    def clear_credentials(self, platform=None):
        """Clear credentials for a platform or all platforms."""
        if platform:
            if platform in self.credentials:
                del self.credentials[platform]
                logger.info(f"Cleared credentials for {platform}")
        else:
            self.credentials = {}
            logger.info("Cleared all credentials")
        
        self.save_credentials()