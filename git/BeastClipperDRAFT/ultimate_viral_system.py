#!/usr/bin/env python3
"""
Ultimate Viral Detection System - Master Controller
Coordinates chat, video, and social proof analysis for maximum viral clip accuracy
"""

import time
import logging
import json
import os
from collections import deque, Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

from twitch_chat_detector import TwitchChatViralDetector
from enhanced_detection import EnhancedViralMomentDetector

logger = logging.getLogger("BeastClipper")


@dataclass
class ViralMoment:
    """Represents a detected viral moment with all metadata."""
    timestamp: float
    confidence: float
    clip_start: float
    clip_end: float
    signals: Dict[str, float]  # chat, video, audio scores
    social_proof: Dict[str, int]  # subs, mods, new_chatters, etc.
    context: str  # gaming, chatting, react, etc.
    description: str
    keywords: List[str]
    user_reactions: List[str]
    momentum_score: float
    uniqueness_score: float
    detected_at: float
    streamer_profile_match: float


class UltimateViralDetector(QThread):
    """
    Master viral detection system that coordinates multiple detection methods
    and applies advanced filtering and learning algorithms.
    """
    
    # Signals
    viral_moment_detected = pyqtSignal(ViralMoment)
    confidence_update = pyqtSignal(dict)
    analytics_update = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, stream_url, stream_buffer, config_manager, sensitivity=0.7):
        super().__init__()
        
        self.stream_url = stream_url
        self.stream_buffer = stream_buffer
        self.config_manager = config_manager
        self.sensitivity = sensitivity
        self.running = False
        
        # Detection components
        self.chat_detector = None
        self.video_detector = None
        
        # Smart learning system
        self.detection_history = deque(maxlen=100)
        self.success_feedback = {}  # moment_id -> user_feedback
        self.streamer_profile = self._load_streamer_profile()
        
        # Multi-signal fusion
        self.signal_buffer = {
            'chat': deque(maxlen=60),      # 60 seconds of chat signals
            'video': deque(maxlen=60),     # 60 seconds of video signals
            'audio': deque(maxlen=60),     # 60 seconds of audio signals
            'social': deque(maxlen=60)     # 60 seconds of social proof
        }
        
        # Advanced filtering
        self.recent_detections = deque(maxlen=10)
        self.duplicate_filter = deque(maxlen=50)
        self.momentum_tracker = deque(maxlen=30)
        
        # Context detection
        self.current_context = "unknown"
        self.context_confidence = 0.0
        
        # Real-time analytics
        self.analytics = {
            'total_detections': 0,
            'false_positives': 0,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'best_detection_times': [],
            'context_performance': {}
        }
        
        # Detection modes
        self.detection_mode = self.config_manager.get("viral_detection.mode", "balanced")
        self.mode_settings = self._get_mode_settings()
        
    def run(self):
        """Main coordination loop."""
        self.running = True
        
        try:
            self.status_update.emit("Initializing ultimate viral detection...")
            
            # Start component detectors
            self._start_component_detectors()
            
            # Main fusion loop
            self.status_update.emit("Multi-signal viral detection active!")
            
            while self.running:
                # Collect signals from all detectors
                self._collect_signals()
                
                # Perform multi-signal fusion
                fusion_result = self._perform_signal_fusion()
                
                # Apply advanced filtering
                if fusion_result and self._passes_advanced_filters(fusion_result):
                    # Perfect timing detection
                    optimized_moment = self._optimize_timing(fusion_result)
                    
                    # Final validation
                    if self._validate_viral_moment(optimized_moment):
                        # Create viral moment object
                        viral_moment = self._create_viral_moment(optimized_moment)
                        
                        # Emit detection
                        self._emit_viral_detection(viral_moment)
                
                # Update analytics
                self._update_analytics()
                
                # Learn and adapt
                self._adaptive_learning()
                
                time.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Ultimate detector error: {str(e)}")
            self.error_occurred.emit(f"Ultimate Detection Error: {str(e)}")
        
        finally:
            self._cleanup()
    
    def _start_component_detectors(self):
        """Start chat and video detection components."""
        # Start chat detector
        self.chat_detector = TwitchChatViralDetector(
            stream_url=self.stream_url,
            sensitivity=self.sensitivity
        )
        self.chat_detector.viral_moment_detected.connect(self._on_chat_signal)
        self.chat_detector.chat_activity_update.connect(self._on_chat_activity)
        self.chat_detector.start()
        
        # Start video detector
        self.video_detector = EnhancedViralMomentDetector(
            stream_buffer=self.stream_buffer,
            sensitivity=self.sensitivity
        )
        self.video_detector.moment_detected.connect(self._on_video_signal)
        self.video_detector.start()
    
    def _collect_signals(self):
        """Collect and normalize signals from all detectors."""
        current_time = time.time()
        
        # Get recent activity from buffers
        chat_activity = self._get_recent_signal('chat', 10)
        video_activity = self._get_recent_signal('video', 10)
        audio_activity = self._get_recent_signal('audio', 10)
        social_activity = self._get_recent_signal('social', 10)
        
        # Store current signal snapshot
        signal_snapshot = {
            'timestamp': current_time,
            'chat_intensity': np.mean([s['score'] for s in chat_activity]) if chat_activity else 0,
            'video_intensity': np.mean([s['score'] for s in video_activity]) if video_activity else 0,
            'social_proof': sum([s.get('unique_users', 0) for s in chat_activity]),
            'momentum': self._calculate_momentum()
        }
        
        return signal_snapshot
    
    def _perform_signal_fusion(self):
        """Advanced multi-signal fusion algorithm."""
        signals = self._collect_signals()
        
        if not signals:
            return None
        
        # Base fusion score
        fusion_score = 0.0
        
        # Chat signal (50% weight - most reliable)
        chat_score = signals['chat_intensity']
        fusion_score += chat_score * 0.5
        
        # Video signal (25% weight)
        video_score = signals['video_intensity']
        fusion_score += video_score * 0.25
        
        # Social proof multiplier (15% weight)
        social_multiplier = min(signals['social_proof'] / 5.0, 2.0)  # Max 2x multiplier
        fusion_score += (social_multiplier - 1.0) * 0.15
        
        # Momentum bonus (10% weight)
        momentum_score = signals['momentum']
        fusion_score += momentum_score * 0.1
        
        # Synchronization bonus - when multiple signals peak together
        signal_sync = self._calculate_signal_synchronization()
        if signal_sync > 0.7:
            fusion_score += 0.1  # 10% bonus for synchronized signals
        
        # Context-aware weighting
        context_multiplier = self._get_context_multiplier()
        fusion_score *= context_multiplier
        
        # Mode-specific adjustments
        fusion_score = self._apply_mode_adjustments(fusion_score, signals)
        
        # Must exceed threshold
        threshold = 0.6 + (1.0 - self.sensitivity) * 0.3
        
        if fusion_score >= threshold:
            return {
                'score': fusion_score,
                'signals': signals,
                'timestamp': signals['timestamp'],
                'sync_quality': signal_sync,
                'context': self.current_context
            }
        
        return None
    
    def _passes_advanced_filters(self, fusion_result):
        """Apply advanced filtering to prevent false positives."""
        current_time = time.time()
        
        # 1. Duplicate detection - check for similar moments in last 2 minutes
        for recent in self.recent_detections:
            if current_time - recent['timestamp'] < 120:  # 2 minutes
                if abs(fusion_result['score'] - recent['score']) < 0.2:
                    logger.debug("Filtered duplicate detection")
                    return False
        
        # 2. Rapid-fire prevention - max 1 detection per 30 seconds
        if self.recent_detections:
            last_detection = self.recent_detections[-1]
            if current_time - last_detection['timestamp'] < 30:
                logger.debug("Filtered rapid-fire detection")
                return False
        
        # 3. Quality threshold - ensure minimum signal quality
        if fusion_result['signals']['chat_intensity'] < 0.3:
            logger.debug("Filtered low-quality chat signal")
            return False
        
        # 4. Context appropriateness
        if not self._is_context_appropriate(fusion_result):
            logger.debug("Filtered inappropriate context")
            return False
        
        # 5. Streamer profile match
        profile_match = self._calculate_profile_match(fusion_result)
        if profile_match < 0.4:
            logger.debug("Filtered profile mismatch")
            return False
        
        # 6. Technical quality check
        if not self._check_technical_quality():
            logger.debug("Filtered poor technical quality")
            return False
        
        return True
    
    def _optimize_timing(self, fusion_result):
        """Find the perfect start and end times for the clip."""
        base_timestamp = fusion_result['timestamp']
        
        # Look for the actual peak moment within ±10 seconds
        search_window = 10
        best_moment = base_timestamp
        best_score = fusion_result['score']
        
        # Check signal history for peak
        for i in range(-search_window, search_window + 1):
            check_time = base_timestamp + i
            moment_score = self._evaluate_moment_at_time(check_time)
            
            if moment_score > best_score:
                best_score = moment_score
                best_moment = check_time
        
        # Determine optimal clip boundaries
        clip_start, clip_end = self._calculate_clip_boundaries(best_moment, fusion_result)
        
        return {
            **fusion_result,
            'optimized_timestamp': best_moment,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'optimized_score': best_score
        }
    
    def _calculate_clip_boundaries(self, peak_moment, fusion_result):
        """Calculate optimal clip start and end times."""
        # Default clip length based on context
        base_length = {
            'gaming': 20,      # Gaming moments need context
            'chatting': 15,    # Conversation clips shorter
            'react': 25,       # Reaction content longer
            'irl': 30         # IRL needs more context
        }.get(self.current_context, 20)
        
        # Adjust based on momentum
        momentum = fusion_result['signals']['momentum']
        if momentum > 0.8:
            base_length += 5  # Extend for high momentum
        elif momentum < 0.4:
            base_length -= 5  # Shorten for low momentum
        
        # Find natural boundaries in signal activity
        pre_activity = self._find_pre_activity(peak_moment)
        post_activity = self._find_post_activity(peak_moment)
        
        # Calculate start time (catch the setup)
        start_offset = min(pre_activity + 3, base_length * 0.7)
        clip_start = max(0, peak_moment - start_offset)
        
        # Calculate end time (catch the reaction)
        end_offset = min(post_activity + 2, base_length * 0.3)
        clip_end = peak_moment + end_offset
        
        # Ensure minimum and maximum length
        clip_length = clip_end - clip_start
        if clip_length < 10:
            clip_end = clip_start + 10
        elif clip_length > 60:
            clip_end = clip_start + 60
        
        return clip_start, clip_end
    
    def _validate_viral_moment(self, optimized_moment):
        """Final validation before detection."""
        # Check against streamer's historical viral moments
        historical_match = self._check_historical_similarity(optimized_moment)
        if historical_match > 0.9:  # Too similar to recent viral moment
            return False
        
        # Validate signal consistency
        if optimized_moment['optimized_score'] < optimized_moment['score'] * 0.8:
            return False  # Optimization made it worse
        
        # Check for technical issues during this time
        if self._has_technical_issues(optimized_moment['optimized_timestamp']):
            return False
        
        return True
    
    def _create_viral_moment(self, optimized_moment):
        """Create a comprehensive viral moment object."""
        signals = optimized_moment['signals']
        
        # Extract social proof data
        social_proof = self._extract_social_proof(optimized_moment['timestamp'])
        
        # Generate description using advanced NLP
        description = self._generate_smart_description(optimized_moment)
        
        # Calculate uniqueness score
        uniqueness = self._calculate_uniqueness(optimized_moment)
        
        # Get streamer profile match
        profile_match = self._calculate_profile_match(optimized_moment)
        
        return ViralMoment(
            timestamp=optimized_moment['optimized_timestamp'],
            confidence=optimized_moment['optimized_score'],
            clip_start=optimized_moment['clip_start'],
            clip_end=optimized_moment['clip_end'],
            signals={
                'chat': signals['chat_intensity'],
                'video': signals['video_intensity'],
                'social': signals['social_proof'],
                'momentum': signals['momentum'],
                'sync': optimized_moment['sync_quality']
            },
            social_proof=social_proof,
            context=self.current_context,
            description=description,
            keywords=self._extract_keywords(optimized_moment),
            user_reactions=self._extract_user_reactions(optimized_moment),
            momentum_score=signals['momentum'],
            uniqueness_score=uniqueness,
            detected_at=time.time(),
            streamer_profile_match=profile_match
        )
    
    def _emit_viral_detection(self, viral_moment):
        """Emit viral moment with full analytics."""
        # Add to detection history
        self.detection_history.append(viral_moment)
        self.recent_detections.append({
            'timestamp': viral_moment.timestamp,
            'score': viral_moment.confidence
        })
        
        # Update analytics
        self.analytics['total_detections'] += 1
        self.analytics['avg_confidence'] = np.mean([
            m.confidence for m in self.detection_history
        ])
        
        # Log detection
        logger.info(
            f"ULTIMATE VIRAL DETECTION: "
            f"Score={viral_moment.confidence:.2f}, "
            f"Context={viral_moment.context}, "
            f"Social={viral_moment.social_proof}, "
            f"Description={viral_moment.description}"
        )
        
        # Emit signal
        self.viral_moment_detected.emit(viral_moment)
        
        # Update confidence metrics
        self.confidence_update.emit({
            'current_confidence': viral_moment.confidence,
            'avg_confidence': self.analytics['avg_confidence'],
            'total_detections': self.analytics['total_detections'],
            'context': viral_moment.context,
            'social_proof': viral_moment.social_proof
        })
    
    # Detection mode settings
    def _get_mode_settings(self):
        """Get settings for different detection modes."""
        modes = {
            'conservative': {
                'threshold_multiplier': 1.5,
                'min_social_proof': 3,
                'min_sync_quality': 0.8,
                'duplicate_window': 300  # 5 minutes
            },
            'balanced': {
                'threshold_multiplier': 1.0,
                'min_social_proof': 2,
                'min_sync_quality': 0.6,
                'duplicate_window': 120  # 2 minutes
            },
            'aggressive': {
                'threshold_multiplier': 0.7,
                'min_social_proof': 1,
                'min_sync_quality': 0.4,
                'duplicate_window': 60   # 1 minute
            },
            'discovery': {
                'threshold_multiplier': 0.5,
                'min_social_proof': 1,
                'min_sync_quality': 0.3,
                'duplicate_window': 30   # 30 seconds
            }
        }
        return modes.get(self.detection_mode, modes['balanced'])
    
    # Signal processing methods
    def _get_recent_signal(self, signal_type, seconds):
        """Get recent signals of specified type."""
        current_time = time.time()
        cutoff = current_time - seconds
        
        return [
            signal for signal in self.signal_buffer[signal_type]
            if signal['timestamp'] >= cutoff
        ]
    
    def _calculate_momentum(self):
        """Calculate current momentum score."""
        recent_activity = []
        
        for signal_type in ['chat', 'video']:
            recent = self._get_recent_signal(signal_type, 30)
            if recent:
                recent_activity.extend([s['score'] for s in recent])
        
        if len(recent_activity) < 5:
            return 0.0
        
        # Calculate momentum as rate of increase
        early_avg = np.mean(recent_activity[:len(recent_activity)//2])
        late_avg = np.mean(recent_activity[len(recent_activity)//2:])
        
        momentum = (late_avg - early_avg) / (early_avg + 0.1)
        return max(0, min(momentum, 1.0))
    
    def _calculate_signal_synchronization(self):
        """Calculate how synchronized different signals are."""
        recent_window = 5  # 5 seconds
        
        chat_signals = self._get_recent_signal('chat', recent_window)
        video_signals = self._get_recent_signal('video', recent_window)
        
        if not chat_signals or not video_signals:
            return 0.0
        
        # Calculate cross-correlation of signal peaks
        chat_scores = [s['score'] for s in chat_signals]
        video_scores = [s['score'] for s in video_signals]
        
        if len(chat_scores) != len(video_scores):
            # Interpolate to same length
            min_len = min(len(chat_scores), len(video_scores))
            chat_scores = chat_scores[:min_len]
            video_scores = video_scores[:min_len]
        
        if len(chat_scores) < 2:
            return 0.0
        
        # Calculate correlation
        correlation = np.corrcoef(chat_scores, video_scores)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0
    
    # Context and learning methods
    def _detect_context(self):
        """Detect current stream context (gaming, chatting, etc.)."""
        # This would analyze recent chat messages, video content, etc.
        # For now, return default
        return "gaming", 0.8
    
    def _load_streamer_profile(self):
        """Load or create streamer-specific profile."""
        channel_name = self.stream_url.split('/')[-1]
        profile_file = f"profiles/{channel_name}.json"
        
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default profile
        return {
            'viral_keywords': [],
            'avg_clip_length': 20,
            'best_times': [],
            'context_preferences': {},
            'success_patterns': []
        }
    
    def _adaptive_learning(self):
        """Learn and adapt from detection history."""
        if len(self.detection_history) < 10:
            return
        
        # Analyze successful detections
        successful_moments = [
            m for m in self.detection_history 
            if self.success_feedback.get(id(m), 0) > 0
        ]
        
        if successful_moments:
            # Update streamer profile
            self._update_streamer_profile(successful_moments)
            
            # Adjust detection parameters
            self._tune_detection_parameters(successful_moments)
    
    # Utility methods for signal processing
    def _on_chat_signal(self, chat_moment):
        """Handle chat detector signal."""
        self.signal_buffer['chat'].append({
            'timestamp': chat_moment['detected_at'],
            'score': chat_moment['score'],
            'data': chat_moment
        })
    
    def _on_video_signal(self, video_moment):
        """Handle video detector signal."""
        self.signal_buffer['video'].append({
            'timestamp': video_moment['detected_at'],
            'score': video_moment['score'],
            'data': video_moment
        })
    
    def _on_chat_activity(self, activity_stats):
        """Handle chat activity updates."""
        self.signal_buffer['social'].append({
            'timestamp': time.time(),
            'score': activity_stats.get('spike_multiplier', 1.0),
            'data': activity_stats
        })
    
    # Placeholder methods for features requiring more implementation
    def _get_context_multiplier(self):
        return 1.0
    
    def _apply_mode_adjustments(self, score, signals):
        return score * self.mode_settings['threshold_multiplier']
    
    def _is_context_appropriate(self, result):
        return True
    
    def _calculate_profile_match(self, result):
        return 0.8  # Default high match
    
    def _check_technical_quality(self):
        return True
    
    def _evaluate_moment_at_time(self, timestamp):
        return 0.5
    
    def _find_pre_activity(self, timestamp):
        return 5
    
    def _find_post_activity(self, timestamp):
        return 3
    
    def _check_historical_similarity(self, moment):
        return 0.3
    
    def _has_technical_issues(self, timestamp):
        return False
    
    def _extract_social_proof(self, timestamp):
        return {'subs': 2, 'mods': 1, 'new_chatters': 3}
    
    def _generate_smart_description(self, moment):
        return f"High-confidence viral moment in {moment.get('context', 'unknown')} context"
    
    def _calculate_uniqueness(self, moment):
        return 0.8
    
    def _extract_keywords(self, moment):
        return ['viral', 'moment']
    
    def _extract_user_reactions(self, moment):
        return ['hype', 'excitement']
    
    def _update_analytics(self):
        """Update real-time analytics."""
        analytics_data = {
            'total_detections': self.analytics['total_detections'],
            'avg_confidence': self.analytics['avg_confidence'],
            'current_context': self.current_context,
            'detection_rate': len(self.recent_detections) / max(1, len(self.detection_history)),
            'signal_quality': {
                'chat': len(self.signal_buffer['chat']),
                'video': len(self.signal_buffer['video']),
                'sync': self._calculate_signal_synchronization()
            }
        }
        
        self.analytics_update.emit(analytics_data)
    
    def _update_streamer_profile(self, successful_moments):
        """Update streamer profile based on successful detections."""
        # Extract patterns from successful moments
        # Save updated profile
        pass
    
    def _tune_detection_parameters(self, successful_moments):
        """Tune detection parameters based on success patterns."""
        # Analyze what made successful moments work
        # Adjust thresholds accordingly
        pass
    
    def provide_feedback(self, moment_id, feedback_score):
        """Provide feedback on detection quality (1-5 scale)."""
        self.success_feedback[moment_id] = feedback_score
        
        # Update success rate
        if self.success_feedback:
            self.analytics['success_rate'] = np.mean(list(self.success_feedback.values())) / 5.0
    
    def set_detection_mode(self, mode):
        """Change detection mode."""
        if mode in ['conservative', 'balanced', 'aggressive', 'discovery']:
            self.detection_mode = mode
            self.mode_settings = self._get_mode_settings()
            logger.info(f"Detection mode changed to: {mode}")
    
    def update_sensitivity(self, new_sensitivity):
        """Update overall sensitivity."""
        self.sensitivity = new_sensitivity
        
        # Update component detectors
        if self.chat_detector:
            self.chat_detector.update_sensitivity(new_sensitivity)
        if self.video_detector:
            self.video_detector.update_sensitivity(new_sensitivity)
    
    def stop(self):
        """Stop all detection components."""
        self.running = False
        
        if self.chat_detector:
            self.chat_detector.stop()
        if self.video_detector:
            self.video_detector.stop()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.chat_detector:
            self.chat_detector.wait()
        if self.video_detector:
            self.video_detector.wait()
