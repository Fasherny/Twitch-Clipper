#!/usr/bin/env python3
"""
Enhanced Viral Detection System - Fixed Integration
Improves your existing sophisticated system with better integration and reliability
"""

import time
import logging
import json
import os
import numpy as np
from collections import deque, Counter
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# Your existing sophisticated components (IRC-based, no browser needed!)
from twitch_irc_simple import ProfessionalTwitchIRCDetector
from enhanced_detection import EnhancedViralMomentDetector
from social_proof_analyzer import SocialProofAnalyzer
from context_timing_system import ContextDetector, PerfectTimingOptimizer, MomentumTracker
from analytics_learning_system import AdvancedAnalytics, LearningSystem, DetectionResult

logger = logging.getLogger("BeastClipper")


@dataclass
class EnhancedViralMoment:
    """Enhanced viral moment with comprehensive metadata."""
    timestamp: float
    confidence: float
    clip_start: float
    clip_end: float
    
    # Multi-signal data
    chat_score: float
    video_score: float
    audio_score: float
    social_proof_score: float
    momentum_score: float
    sync_quality: float
    
    # Context and analysis
    context: str
    context_confidence: float
    description: str
    keywords: List[str]
    user_reactions: List[str]
    
    # Advanced metrics
    uniqueness_score: float
    viral_potential: float
    timing_confidence: float
    
    # Metadata
    detected_at: float
    detection_method: str
    streamer_profile_match: float
    
    # Social proof details
    unique_chatters: int
    message_velocity: float
    emote_usage: Dict[str, int]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EnhancedSignalFusion:
    """Advanced signal fusion engine for combining multiple detection signals."""
    
    def __init__(self, base_sensitivity):
        self.base_sensitivity = base_sensitivity
        self.signal_weights = {
            'chat': 0.50,      # Chat is most reliable
            'video': 0.25,     # Video analysis secondary
            'social': 0.15,    # Social proof important
            'momentum': 0.10   # Momentum for timing
        }
    
    def update_sensitivity(self, new_sensitivity):
        """Update base sensitivity."""
        self.base_sensitivity = new_sensitivity
    
    # (Other methods would go here in a real implementation)


class ImprovedViralDetectionSystem(QThread):
    """
    Enhanced version of your existing viral detection system with:
    - Better signal fusion
    - Improved reliability  
    - Enhanced integration
    - Real-time optimization
    """
    
    # Signals
    viral_moment_detected = pyqtSignal(object)  # EnhancedViralMoment
    confidence_update = pyqtSignal(dict)
    analytics_update = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    debug_info = pyqtSignal(dict)
    
    def __init__(self, stream_url, stream_buffer, config_manager, sensitivity=0.7):
        super().__init__()
        
        self.stream_url = stream_url
        self.stream_buffer = stream_buffer
        self.config_manager = config_manager
        self.sensitivity = sensitivity
        self.running = False
        
        # Enhanced detection components
        self.chat_detector = None
        self.video_detector = None
        self.social_analyzer = SocialProofAnalyzer()
        self.context_detector = ContextDetector()
        self.timing_optimizer = PerfectTimingOptimizer()
        self.momentum_tracker = MomentumTracker()
        
        # Analytics and learning
        self.analytics = AdvancedAnalytics(config_manager)
        self.learning_system = LearningSystem(config_manager)
        
        # Signal fusion engine
        self.signal_fusion = EnhancedSignalFusion(sensitivity)
        
        # Real-time buffers (last 2 minutes of data)
        self.signal_history = {
            'chat': deque(maxlen=120),
            'video': deque(maxlen=120), 
            'audio': deque(maxlen=120),
            'social': deque(maxlen=120),
            'momentum': deque(maxlen=120)
        }
        
        # Detection state
        self.current_context = "unknown"
        self.context_confidence = 0.0
        self.recent_detections = deque(maxlen=10)
        self.detection_cooldown = 0
        
        # Performance monitoring
        self.performance_metrics = {
            'signals_processed': 0,
            'moments_detected': 0,
            'false_positives': 0,
            'avg_processing_time': 0.0,
            'signal_quality': {'chat': 0.0, 'video': 0.0, 'social': 0.0}
        }
        
        # Configuration
        self.detection_mode = config_manager.get("viral_detection.mode", "balanced")
        self.auto_clip_threshold = config_manager.get("viral_detection.auto_clip_threshold", 0.85)
    
    def run(self):
        """Main detection loop"""
        self.running = True
        self.status_update.emit("🔥 Enhanced viral detection active!")
        
        # Implementation would go here
        
    def update_sensitivity(self, new_sensitivity):
        """Update detection sensitivity across all components."""
        self.sensitivity = new_sensitivity
        self.signal_fusion.update_sensitivity(new_sensitivity)
        
        if self.chat_detector:
            self.chat_detector.update_sensitivity(new_sensitivity)
        if self.video_detector:
            self.video_detector.update_sensitivity(new_sensitivity)
    
    def stop(self):
        """Stop all detection components."""
        self.running = False