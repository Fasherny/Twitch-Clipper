#!/usr/bin/env python3
"""
Advanced Analytics & Learning System
Provides real-time analytics and learns from user feedback to improve detection
"""

import time
import logging
import json
import os
import pickle
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("BeastClipper")


@dataclass
class DetectionResult:
    """Stores complete information about a detection."""
    timestamp: float
    confidence: float
    context: str
    signals: Dict[str, float]
    social_proof: Dict[str, int]
    clip_start: float
    clip_end: float
    user_feedback: Optional[int] = None  # 1-5 rating
    was_clipped: bool = False
    clip_file: Optional[str] = None
    external_success: Optional[Dict] = None  # YouTube views, etc.


class AdvancedAnalytics:
    """Provides comprehensive analytics on viral detection performance."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.detection_results = deque(maxlen=1000)  # Last 1000 detections
        self.session_stats = {
            'start_time': time.time(),
            'total_detections': 0,
            'clips_created': 0,
            'user_ratings': [],
            'context_performance': defaultdict(list),
            'hourly_performance': defaultdict(list),
            'signal_effectiveness': defaultdict(list)
        }
        
        # Load historical data
        self.historical_data = self._load_historical_data()
        
        # Real-time metrics
        self.real_time_metrics = {
            'current_detection_rate': 0.0,
            'average_confidence': 0.0,
            'success_rate': 0.0,
            'context_distribution': Counter(),
            'signal_quality': {'chat': 0.0, 'video': 0.0, 'social': 0.0}
        }
        
        # Performance tracking
        self.performance_windows = {
            'last_hour': deque(maxlen=60),      # Per-minute stats
            'last_day': deque(maxlen=24),       # Per-hour stats  
            'last_week': deque(maxlen=7),       # Per-day stats
            'last_month': deque(maxlen=30)      # Per-day stats
        }
    
    def record_detection(self, detection_result: DetectionResult):
        """Record a new detection result."""
        self.detection_results.append(detection_result)
        self.session_stats['total_detections'] += 1
        
        # Update real-time metrics
        self._update_real_time_metrics()
        
        # Update context performance
        self.session_stats['context_performance'][detection_result.context].append(detection_result.confidence)
        
        # Update hourly performance
        hour = datetime.fromtimestamp(detection_result.timestamp).hour
        self.session_stats['hourly_performance'][hour].append(detection_result.confidence)
        
        # Update signal effectiveness
        for signal_type, score in detection_result.signals.items():
            self.session_stats['signal_effectiveness'][signal_type].append(score)
        
        logger.info(f"Recorded detection: {detection_result.context} context, confidence {detection_result.confidence:.2f}")
    
    def record_user_feedback(self, detection_id: int, rating: int, was_clipped: bool = False, clip_file: str = None):
        """Record user feedback on a detection."""
        if detection_id < len(self.detection_results):
            detection = self.detection_results[detection_id]
            detection.user_feedback = rating
            detection.was_clipped = was_clipped
            detection.clip_file = clip_file
            
            self.session_stats['user_ratings'].append(rating)
            if was_clipped:
                self.session_stats['clips_created'] += 1
            
            logger.info(f"User feedback recorded: {rating}/5 for detection {detection_id}")
    
    def _update_real_time_metrics(self):
        """Update real-time performance metrics."""
        if not self.detection_results:
            return
        
        recent_detections = [d for d in self.detection_results if time.time() - d.timestamp <= 300]  # Last 5 minutes
        
        # Detection rate (per minute)
        if recent_detections:
            time_span = max(300, time.time() - recent_detections[0].timestamp)
            self.real_time_metrics['current_detection_rate'] = len(recent_detections) / (time_span / 60)
        
        # Average confidence
        if recent_detections:
            self.real_time_metrics['average_confidence'] = np.mean([d.confidence for d in recent_detections])
        
        # Success rate (based on user feedback)
        rated_detections = [d for d in recent_detections if d.user_feedback is not None]
        if rated_detections:
            good_ratings = sum(1 for d in rated_detections if d.user_feedback >= 4)
            self.real_time_metrics['success_rate'] = good_ratings / len(rated_detections)
        
        # Context distribution
        self.real_time_metrics['context_distribution'] = Counter([d.context for d in recent_detections])
        
        # Signal quality
        if recent_detections:
            for signal_type in ['chat', 'video', 'social']:
                scores = [d.signals.get(signal_type, 0) for d in recent_detections]
                self.real_time_metrics['signal_quality'][signal_type] = np.mean(scores)
    
    def get_performance_summary(self, timeframe='session'):
        """Get comprehensive performance summary."""
        if timeframe == 'session':
            return self._get_session_summary()
        elif timeframe == 'historical':
            return self._get_historical_summary()
        else:
            return self._get_windowed_summary(timeframe)
    
    def _get_session_summary(self):
        """Get current session performance summary."""
        session_duration = time.time() - self.session_stats['start_time']
        
        summary = {
            'session_duration_minutes': session_duration / 60,
            'total_detections': self.session_stats['total_detections'],
            'clips_created': self.session_stats['clips_created'],
            'detection_rate_per_hour': (self.session_stats['total_detections'] / session_duration) * 3600,
            'clip_conversion_rate': self.session_stats['clips_created'] / max(self.session_stats['total_detections'], 1),
            'real_time_metrics': self.real_time_metrics.copy()
        }
        
        # User satisfaction
        if self.session_stats['user_ratings']:
            summary['average_user_rating'] = np.mean(self.session_stats['user_ratings'])
            summary['user_satisfaction_rate'] = sum(1 for r in self.session_stats['user_ratings'] if r >= 4) / len(self.session_stats['user_ratings'])
        
        # Context performance
        context_performance = {}
        for context, confidences in self.session_stats['context_performance'].items():
            context_performance[context] = {
                'detections': len(confidences),
                'avg_confidence': np.mean(confidences),
                'best_confidence': max(confidences),
                'consistency': 1 - (np.std(confidences) / max(np.mean(confidences), 0.1))
            }
        summary['context_performance'] = context_performance
        
        # Signal effectiveness
        signal_effectiveness = {}
        for signal_type, scores in self.session_stats['signal_effectiveness'].items():
            signal_effectiveness[signal_type] = {
                'average_score': np.mean(scores),
                'reliability': 1 - (np.std(scores) / max(np.mean(scores), 0.1)),
                'peak_score': max(scores)
            }
        summary['signal_effectiveness'] = signal_effectiveness
        
        return summary
    
    def _get_historical_summary(self):
        """Get historical performance summary."""
        return {
            'total_sessions': len(self.historical_data.get('sessions', [])),
            'lifetime_detections': self.historical_data.get('lifetime_detections', 0),
            'lifetime_clips': self.historical_data.get('lifetime_clips', 0),
            'best_session_performance': self.historical_data.get('best_session', {}),
            'learning_improvements': self._calculate_learning_improvements()
        }
    
    def get_optimization_recommendations(self):
        """Generate optimization recommendations based on analytics."""
        recommendations = []
        
        # Analyze recent performance
        recent_detections = [d for d in self.detection_results if time.time() - d.timestamp <= 1800]  # Last 30 minutes
        
        if not recent_detections:
            return ["Not enough recent data for recommendations"]
        
        # Low success rate
        rated_recent = [d for d in recent_detections if d.user_feedback is not None]
        if rated_recent:
            success_rate = sum(1 for d in rated_recent if d.user_feedback >= 4) / len(rated_recent)
            if success_rate < 0.6:
                recommendations.append("Consider increasing sensitivity threshold - too many false positives")
        
        # Context-specific recommendations
        context_performance = defaultdict(list)
        for detection in recent_detections:
            if detection.user_feedback is not None:
                context_performance[detection.context].append(detection.user_feedback)
        
        for context, ratings in context_performance.items():
            if len(ratings) >= 3:
                avg_rating = np.mean(ratings)
                if avg_rating < 3:
                    recommendations.append(f"Poor performance in {context} context - consider adjusting detection parameters")
                elif avg_rating >= 4.5:
                    recommendations.append(f"Excellent performance in {context} context - current settings are optimal")
        
        # Signal quality recommendations
        signal_scores = self.real_time_metrics['signal_quality']
        if signal_scores['chat'] > signal_scores['video'] * 2:
            recommendations.append("Chat analysis much stronger than video - consider increasing chat weight")
        elif signal_scores['video'] > signal_scores['chat'] * 2:
            recommendations.append("Video analysis much stronger than chat - consider increasing video weight")
        
        # Detection rate recommendations
        detection_rate = self.real_time_metrics['current_detection_rate']
        if detection_rate > 6:  # More than 6 per minute
            recommendations.append("Very high detection rate - consider increasing selectivity")
        elif detection_rate < 0.5:  # Less than 0.5 per minute
            recommendations.append("Low detection rate - consider decreasing selectivity")
        
        return recommendations if recommendations else ["Current performance is well-balanced"]
    
    def _load_historical_data(self):
        """Load historical analytics data."""
        data_file = "analytics_history.json"
        try:
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
        
        return {
            'sessions': [],
            'lifetime_detections': 0,
            'lifetime_clips': 0,
            'best_session': {},
            'learning_data': {}
        }
    
    def save_session_data(self):
        """Save current session data to historical records."""
        session_summary = self._get_session_summary()
        
        # Add to historical data
        self.historical_data['sessions'].append({
            'timestamp': self.session_stats['start_time'],
            'summary': session_summary,
            'detections': [asdict(d) for d in self.detection_results]
        })
        
        # Update lifetime stats
        self.historical_data['lifetime_detections'] += self.session_stats['total_detections']
        self.historical_data['lifetime_clips'] += self.session_stats['clips_created']
        
        # Update best session
        if not self.historical_data['best_session'] or session_summary.get('user_satisfaction_rate', 0) > self.historical_data['best_session'].get('user_satisfaction_rate', 0):
            self.historical_data['best_session'] = session_summary
        
        # Save to file
        try:
            with open("analytics_history.json", 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
    
    def _calculate_learning_improvements(self):
        """Calculate improvements from learning over time."""
        sessions = self.historical_data.get('sessions', [])
        if len(sessions) < 5:
            return "Insufficient data for learning analysis"
        
        # Compare early vs recent sessions
        early_sessions = sessions[:len(sessions)//3]
        recent_sessions = sessions[-len(sessions)//3:]
        
        early_satisfaction = np.mean([s['summary'].get('user_satisfaction_rate', 0) for s in early_sessions])
        recent_satisfaction = np.mean([s['summary'].get('user_satisfaction_rate', 0) for s in recent_sessions])
        
        improvement = recent_satisfaction - early_satisfaction
        
        return {
            'satisfaction_improvement': improvement,
            'improvement_percentage': improvement * 100,
            'trend': 'improving' if improvement > 0.05 else 'stable' if improvement > -0.05 else 'declining'
        }


class LearningSystem:
    """Machine learning system that adapts detection parameters based on feedback."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.feature_weights = {
            'chat_score': 0.5,
            'video_score': 0.25,
            'social_proof': 0.15,
            'momentum': 0.1
        }
        
        self.context_preferences = defaultdict(lambda: {
            'sensitivity_multiplier': 1.0,
            'signal_weights': self.feature_weights.copy(),
            'timing_preferences': {}
        })
        
        self.streamer_profile = {
            'typical_viral_patterns': [],
            'best_detection_times': [],
            'audience_preferences': {},
            'content_style': 'unknown'
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_samples_for_learning = 10
        
        # Load existing model
        self._load_learning_model()
    
    def process_feedback(self, detection_result: DetectionResult):
        """Process user feedback to improve future detections."""
        if detection_result.user_feedback is None:
            return
        
        # Extract features
        features = self._extract_features(detection_result)
        
        # Update context preferences
        self._update_context_preferences(detection_result, features)
        
        # Update feature weights
        self._update_feature_weights(detection_result, features)
        
        # Update streamer profile
        self._update_streamer_profile(detection_result)
        
        # Save updated model
        self._save_learning_model()
    
    def _extract_features(self, detection_result: DetectionResult):
        """Extract features from detection result."""
        return {
            'chat_score': detection_result.signals.get('chat', 0),
            'video_score': detection_result.signals.get('video', 0),
            'social_proof_score': detection_result.signals.get('social', 0),
            'momentum_score': detection_result.signals.get('momentum', 0),
            'sync_quality': detection_result.signals.get('sync', 0),
            'context': detection_result.context,
            'confidence': detection_result.confidence,
            'time_of_day': datetime.fromtimestamp(detection_result.timestamp).hour,
            'clip_duration': detection_result.clip_end - detection_result.clip_start
        }
    
    def _update_context_preferences(self, detection_result: DetectionResult, features):
        """Update preferences for specific contexts."""
        context = detection_result.context
        feedback = detection_result.user_feedback
        
        # Positive feedback (4-5) increases preference, negative (1-2) decreases
        feedback_score = (feedback - 3) / 2  # Maps 1-5 to -1 to +1
        
        # Update sensitivity multiplier
        current_multiplier = self.context_preferences[context]['sensitivity_multiplier']
        if feedback_score > 0:
            # Good detection - slightly increase sensitivity for this context
            new_multiplier = current_multiplier + (self.learning_rate * feedback_score * 0.1)
        else:
            # Bad detection - decrease sensitivity for this context
            new_multiplier = current_multiplier + (self.learning_rate * feedback_score * 0.2)
        
        self.context_preferences[context]['sensitivity_multiplier'] = max(0.5, min(new_multiplier, 2.0))
        
        logger.info(f"Updated {context} sensitivity multiplier to {new_multiplier:.2f}")
    
    def _update_feature_weights(self, detection_result: DetectionResult, features):
        """Update feature weights based on what led to good/bad detections."""
        feedback = detection_result.user_feedback
        feedback_score = (feedback - 3) / 2  # Maps 1-5 to -1 to +1
        
        # Identify which signals were strongest for this detection
        signal_strengths = {
            'chat_score': features['chat_score'],
            'video_score': features['video_score'],
            'social_proof': features['social_proof_score'],
            'momentum': features['momentum_score']
        }
        
        # Update weights based on feedback
        for signal_type, strength in signal_strengths.items():
            if strength > 0.5:  # Only adjust for strong signals
                current_weight = self.feature_weights[signal_type]
                adjustment = self.learning_rate * feedback_score * strength * 0.05
                new_weight = current_weight + adjustment
                self.feature_weights[signal_type] = max(0.05, min(new_weight, 0.8))
        
        # Normalize weights to sum to 1
        total_weight = sum(self.feature_weights.values())
        for signal_type in self.feature_weights:
            self.feature_weights[signal_type] /= total_weight
    
    def _update_streamer_profile(self, detection_result: DetectionResult):
        """Update streamer-specific profile."""
        if detection_result.user_feedback >= 4:  # Good detection
            # Add to typical viral patterns
            pattern = {
                'context': detection_result.context,
                'signals': detection_result.signals,
                'social_proof': detection_result.social_proof,
                'time_of_day': datetime.fromtimestamp(detection_result.timestamp).hour
            }
            
            self.streamer_profile['typical_viral_patterns'].append(pattern)
            
            # Keep only recent patterns (last 50)
            if len(self.streamer_profile['typical_viral_patterns']) > 50:
                self.streamer_profile['typical_viral_patterns'].pop(0)
            
            # Update best detection times
            hour = datetime.fromtimestamp(detection_result.timestamp).hour
            self.streamer_profile['best_detection_times'].append(hour)
            
            # Keep only recent times (last 100)
            if len(self.streamer_profile['best_detection_times']) > 100:
                self.streamer_profile['best_detection_times'].pop(0)
    
    def get_adjusted_parameters(self, context, base_sensitivity):
        """Get adjusted parameters for a specific context."""
        context_prefs = self.context_preferences[context]
        
        adjusted_sensitivity = base_sensitivity * context_prefs['sensitivity_multiplier']
        adjusted_weights = context_prefs.get('signal_weights', self.feature_weights)
        
        return {
            'sensitivity': max(0.1, min(adjusted_sensitivity, 1.0)),
            'feature_weights': adjusted_weights,
            'context_multiplier': context_prefs['sensitivity_multiplier']
        }
    
    def predict_viral_potential(self, features):
        """Predict viral potential of a moment based on learned patterns."""
        # Simple scoring based on feature weights
        score = 0.0
        
        for feature, weight in self.feature_weights.items():
            feature_value = features.get(feature, 0)
            score += feature_value * weight
        
        # Context adjustment
        context = features.get('context', 'unknown')
        context_multiplier = self.context_preferences[context]['sensitivity_multiplier']
        score *= context_multiplier
        
        # Time-of-day adjustment
        hour = features.get('time_of_day', 12)
        if self.streamer_profile['best_detection_times']:
            hour_frequency = self.streamer_profile['best_detection_times'].count(hour)
            total_detections = len(self.streamer_profile['best_detection_times'])
            hour_multiplier = 1.0 + (hour_frequency / total_detections - 1/24) * 2  # Boost good hours
            score *= hour_multiplier
        
        return min(score, 1.0)
    
    def get_learning_insights(self):
        """Get insights about what the system has learned."""
        insights = []
        
        # Feature importance insights
        sorted_features = sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)
        insights.append(f"Most important signal: {sorted_features[0][0]} ({sorted_features[0][1]:.2f})")
        
        # Context insights - ADD SAFETY CHECK
        if self.context_preferences:  # Check if not empty
            best_context = max(self.context_preferences.items(), key=lambda x: x[1]['sensitivity_multiplier'])
            worst_context = min(self.context_preferences.items(), key=lambda x: x[1]['sensitivity_multiplier'])
            
            if best_context[1]['sensitivity_multiplier'] > 1.1:
                insights.append(f"Best performing context: {best_context[0]}")
            if worst_context[1]['sensitivity_multiplier'] < 0.9:
                insights.append(f"Challenging context: {worst_context[0]}")
        else:
            insights.append("Learning context patterns - insufficient data yet")
        
        # Time insights
        if self.streamer_profile['best_detection_times']:
            best_hours = Counter(self.streamer_profile['best_detection_times']).most_common(3)
            insights.append(f"Best detection hours: {[f'{h}:00' for h, _ in best_hours]}")
        else:
            insights.append("Analyzing optimal detection times - need more data")
        
        return insights
    
    def _load_learning_model(self):
        """Load existing learning model."""
        model_file = "learning_model.pkl"
        try:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.feature_weights = data.get('feature_weights', self.feature_weights)
                    self.context_preferences = data.get('context_preferences', self.context_preferences)
                    self.streamer_profile = data.get('streamer_profile', self.streamer_profile)
                logger.info("Loaded existing learning model")
        except Exception as e:
            logger.error(f"Error loading learning model: {e}")
    
    def _save_learning_model(self):
        """Save current learning model."""
        model_file = "learning_model.pkl"
        try:
            data = {
                'feature_weights': dict(self.feature_weights),
                'context_preferences': dict(self.context_preferences),
                'streamer_profile': self.streamer_profile,
                'last_updated': time.time()
            }
            with open(model_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving learning model: {e}")


class ExternalValidationSystem:
    """System for validating clips against external metrics."""
    
    def __init__(self):
        self.validation_sources = {
            'reddit_lsf': {'enabled': False, 'api_key': None},
            'twitter_trending': {'enabled': False, 'api_key': None},
            'youtube_analytics': {'enabled': False, 'api_key': None},
            'tiktok_metrics': {'enabled': False, 'api_key': None}
        }
        
        self.validation_history = deque(maxlen=100)
    
    def validate_clip_external(self, clip_file, keywords):
        """Validate a clip against external viral metrics."""
        # This would implement actual API calls to various platforms
        # For now, return mock validation
        
        validation_result = {
            'timestamp': time.time(),
            'clip_file': clip_file,
            'keywords': keywords,
            'external_scores': {
                'reddit_mentions': 0,
                'twitter_mentions': 0,
                'youtube_views': 0,
                'tiktok_shares': 0
            },
            'viral_prediction': 0.0
        }
        
        # Mock some validation logic
        if any(keyword in ['insane', 'crazy', 'wtf', 'clip'] for keyword in keywords):
            validation_result['viral_prediction'] = 0.7
        else:
            validation_result['viral_prediction'] = 0.3
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def get_trending_topics(self):
        """Get current trending topics related to streaming."""
        # This would fetch actual trending data
        return {
            'gaming_trends': ['new_game_release', 'tournament_results'],
            'general_trends': ['viral_meme', 'current_event'],
            'platform_trends': ['twitch_feature', 'streamer_drama']
        }