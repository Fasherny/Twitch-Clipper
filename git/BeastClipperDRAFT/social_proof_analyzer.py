#!/usr/bin/env python3
"""
Social Proof & User Intelligence System
Analyzes chat user types and social validation signals for viral detection
"""

import time
import logging
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger("BeastClipper")


class SocialProofAnalyzer:
    """Analyzes social proof signals from Twitch chat users."""
    
    def __init__(self):
        self.user_database = {}  # username -> user_profile
        self.message_history = []
        self.social_signals = {
            'subscriber_reactions': [],
            'moderator_reactions': [],
            'vip_reactions': [],
            'new_chatter_reactions': [],
            'lurker_activations': [],
            'follower_notifications': []
        }
        
        # User type detection patterns
        self.badge_patterns = {
            'subscriber': ['subscriber', 'sub'],
            'moderator': ['moderator', 'mod'],
            'vip': ['vip'],
            'broadcaster': ['broadcaster'],
            'staff': ['staff', 'admin'],
            'turbo': ['turbo'],
            'prime': ['prime']
        }
        
        # Behavioral patterns for user classification
        self.behavioral_indicators = {
            'regular_chatter': {'min_messages': 10, 'time_span_days': 7},
            'lurker': {'max_messages': 3, 'time_span_days': 30},
            'new_viewer': {'max_messages': 5, 'first_seen_hours': 24},
            'community_leader': {'min_messages': 50, 'positive_reactions': 20}
        }
    
    def analyze_message_social_proof(self, message_data):
        """
        Analyze a single message for social proof indicators.
        
        Args:
            message_data: Dict with 'username', 'text', 'badges', 'timestamp'
        
        Returns:
            Dict with social proof scores and classifications
        """
        username = message_data['username']
        text = message_data['text']
        badges = message_data.get('badges', [])
        timestamp = message_data['timestamp']
        
        # Update user profile
        self._update_user_profile(username, message_data)
        
        # Get user classification
        user_type = self._classify_user(username)
        
        # Calculate social proof weight
        proof_weight = self._calculate_proof_weight(user_type, badges)
        
        # Detect special reaction types
        reaction_type = self._detect_reaction_type(text)
        
        # Check for viral indicators
        viral_indicators = self._extract_viral_indicators(text, user_type)
        
        social_proof = {
            'username': username,
            'user_type': user_type,
            'proof_weight': proof_weight,
            'reaction_type': reaction_type,
            'viral_indicators': viral_indicators,
            'badges': badges,
            'is_first_time_speaker': self._is_first_time_speaker(username),
            'influence_score': self._calculate_influence_score(username),
            'authenticity_score': self._calculate_authenticity_score(username, text)
        }
        
        # Track special events
        self._track_special_events(social_proof)
        
        return social_proof
    
    def _update_user_profile(self, username, message_data):
        """Update user profile with new message data."""
        if username not in self.user_database:
            self.user_database[username] = {
                'first_seen': message_data['timestamp'],
                'message_count': 0,
                'total_characters': 0,
                'badges_seen': set(),
                'reaction_history': [],
                'viral_moment_participation': 0,
                'last_active': message_data['timestamp'],
                'typical_message_length': 0,
                'emoji_usage': 0,
                'caps_usage': 0
            }
        
        profile = self.user_database[username]
        
        # Update stats
        profile['message_count'] += 1
        profile['total_characters'] += len(message_data['text'])
        profile['last_active'] = message_data['timestamp']
        profile['badges_seen'].update(message_data.get('badges', []))
        
        # Update message patterns
        text = message_data['text']
        profile['typical_message_length'] = (
            (profile['typical_message_length'] * (profile['message_count'] - 1) + len(text)) 
            / profile['message_count']
        )
        
        # Track emoji and caps usage
        emoji_count = len(re.findall(r'[😀-🙏]', text))
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        profile['emoji_usage'] = (
            (profile['emoji_usage'] * (profile['message_count'] - 1) + emoji_count) 
            / profile['message_count']
        )
        
        profile['caps_usage'] = (
            (profile['caps_usage'] * (profile['message_count'] - 1) + caps_ratio) 
            / profile['message_count']
        )
    
    def _classify_user(self, username):
        """Classify user based on their behavior and badges."""
        if username not in self.user_database:
            return 'unknown'
        
        profile = self.user_database[username]
        current_time = time.time()
        
        # Check badges first
        badges = profile['badges_seen']
        if 'broadcaster' in badges:
            return 'broadcaster'
        elif 'moderator' in badges or 'mod' in badges:
            return 'moderator'
        elif 'vip' in badges:
            return 'vip'
        elif 'subscriber' in badges or 'sub' in badges:
            return 'subscriber'
        elif 'staff' in badges or 'admin' in badges:
            return 'staff'
        
        # Behavioral classification
        days_since_first = (current_time - profile['first_seen']) / 86400
        
        if days_since_first < 1:
            return 'new_viewer'
        elif profile['message_count'] <= 3 and days_since_first >= 7:
            return 'lurker'
        elif profile['message_count'] >= 50 and profile['viral_moment_participation'] >= 5:
            return 'community_leader'
        elif profile['message_count'] >= 10 and days_since_first >= 7:
            return 'regular_chatter'
        else:
            return 'casual_viewer'
    
    def _calculate_proof_weight(self, user_type, badges):
        """Calculate social proof weight based on user type and badges."""
        # Base weights by user type
        base_weights = {
            'broadcaster': 10.0,    # Streamer's own reactions are very important
            'moderator': 5.0,       # Mods know the community
            'staff': 4.0,          # Twitch staff
            'vip': 3.0,            # VIPs are trusted community members  
            'subscriber': 2.5,      # Paying supporters
            'community_leader': 2.0, # Active positive community members
            'regular_chatter': 1.5,  # Known regulars
            'casual_viewer': 1.0,    # Standard weight
            'new_viewer': 1.2,       # New viewers speaking up is interesting
            'lurker': 3.0,          # Lurkers speaking up is very significant
            'unknown': 0.8          # Unknown users slightly discounted
        }
        
        weight = base_weights.get(user_type, 1.0)
        
        # Badge multipliers
        if 'turbo' in badges:
            weight *= 1.2
        if 'prime' in badges:
            weight *= 1.1
        
        return weight
    
    def _detect_reaction_type(self, text):
        """Detect the type of reaction from message text."""
        text_lower = text.lower()
        
        # Direct clip requests (highest value)
        if any(phrase in text_lower for phrase in ['clip that', 'clip it', 'someone clip']):
            return 'clip_request'
        
        # Extreme reactions
        if any(word in text_lower for word in ['omg', 'wtf', 'no way', 'insane', 'crazy']):
            return 'shock'
        
        # Laughter
        if any(word in text_lower for word in ['lmao', 'lmfao', 'dying', 'dead', 'crying']):
            return 'laughter'
        
        # Hype/excitement
        if any(word in text_lower for word in ['hype', 'lets go', 'poggers', 'fire', 'sheesh']):
            return 'hype'
        
        # Agreement/validation
        if any(word in text_lower for word in ['facts', 'true', 'real', 'based', 'valid']):
            return 'agreement'
        
        # Question/confusion
        if '?' in text and len([c for c in text if c == '?']) >= 2:
            return 'confusion'
        
        # All caps (excitement)
        if len(text) > 3 and text.isupper():
            return 'excitement'
        
        return 'neutral'
    
    def _extract_viral_indicators(self, text, user_type):
        """Extract specific viral indicators from message."""
        indicators = []
        text_lower = text.lower()
        
        # Direct viral requests
        if 'clip' in text_lower:
            indicators.append('clip_worthy')
        
        # Excitement indicators
        if '!' in text and len([c for c in text if c == '!']) >= 3:
            indicators.append('high_excitement')
        
        # Surprise indicators  
        if any(word in text_lower for word in ['no way', 'bruh', 'what', 'how']):
            indicators.append('surprise')
        
        # Community validation
        if user_type in ['moderator', 'vip', 'community_leader'] and any(
            word in text_lower for word in ['good', 'great', 'amazing', 'perfect']
        ):
            indicators.append('community_validation')
        
        # Timing indicators
        if any(phrase in text_lower for phrase in ['right now', 'just happened', 'did you see']):
            indicators.append('timing_emphasis')
        
        return indicators
    
    def _is_first_time_speaker(self, username):
        """Check if this is user's first time speaking recently."""
        if username not in self.user_database:
            return True
        
        profile = self.user_database[username]
        
        # Consider "first time" if they haven't spoken in the last hour
        return (time.time() - profile['last_active']) > 3600
    
    def _calculate_influence_score(self, username):
        """Calculate user's influence score in the community."""
        if username not in self.user_database:
            return 0.0
        
        profile = self.user_database[username]
        
        # Base score from participation
        base_score = min(profile['viral_moment_participation'] * 0.1, 1.0)
        
        # Message frequency factor
        days_active = max((time.time() - profile['first_seen']) / 86400, 1)
        message_frequency = profile['message_count'] / days_active
        frequency_score = min(message_frequency * 0.05, 0.5)
        
        # Badge bonus
        badge_bonus = len(profile['badges_seen']) * 0.1
        
        return min(base_score + frequency_score + badge_bonus, 2.0)
    
    def _calculate_authenticity_score(self, username, text):
        """Calculate how authentic/human the message appears."""
        if username not in self.user_database:
            return 0.5
        
        profile = self.user_database[username]
        
        # Check for bot-like patterns
        authenticity = 1.0
        
        # Very consistent message length might indicate bot
        if profile['message_count'] > 5:
            length_variance = abs(len(text) - profile['typical_message_length'])
            if length_variance < 2:  # Very consistent lengths
                authenticity -= 0.2
        
        # Check for spam patterns
        if re.search(r'(.)\1{4,}', text):  # Repeated characters
            authenticity -= 0.3
        
        # Very high caps usage might indicate bot
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.8 and len(text) > 10:
            authenticity -= 0.2
        
        # URL presence
        if re.search(r'http[s]?://', text):
            authenticity -= 0.4
        
        return max(authenticity, 0.1)
    
    def _track_special_events(self, social_proof):
        """Track special social events."""
        username = social_proof['username']
        user_type = social_proof['user_type']
        
        # Lurker activation
        if user_type == 'lurker' and social_proof['is_first_time_speaker']:
            self.social_signals['lurker_activations'].append({
                'username': username,
                'timestamp': time.time(),
                'proof_weight': social_proof['proof_weight']
            })
        
        # Important user reactions
        if user_type in ['moderator', 'vip', 'subscriber'] and social_proof['reaction_type'] != 'neutral':
            signal_key = f"{user_type}_reactions"
            if signal_key in self.social_signals:
                self.social_signals[signal_key].append({
                    'username': username,
                    'reaction_type': social_proof['reaction_type'],
                    'timestamp': time.time(),
                    'proof_weight': social_proof['proof_weight']
                })
    
    def get_recent_social_proof_summary(self, seconds=30):
        """Get summary of recent social proof signals."""
        current_time = time.time()
        cutoff = current_time - seconds
        
        summary = {
            'total_proof_weight': 0.0,
            'unique_users': set(),
            'reaction_types': Counter(),
            'user_types': Counter(),
            'viral_indicators': Counter(),
            'special_events': 0
        }
        
        # Analyze messages from the specified time period
        for message in self.message_history:
            if message['timestamp'] >= cutoff:
                proof = self.analyze_message_social_proof(message)
                
                summary['total_proof_weight'] += proof['proof_weight']
                summary['unique_users'].add(proof['username'])
                summary['reaction_types'][proof['reaction_type']] += 1
                summary['user_types'][proof['user_type']] += 1
                
                for indicator in proof['viral_indicators']:
                    summary['viral_indicators'][indicator] += 1
        
        # Count special events
        for event_type, events in self.social_signals.items():
            recent_events = [e for e in events if e['timestamp'] >= cutoff]
            summary['special_events'] += len(recent_events)
        
        # Convert unique_users set to count
        summary['unique_users'] = len(summary['unique_users'])
        
        return summary
    
    def calculate_viral_social_score(self, seconds=30):
        """Calculate overall viral social proof score for recent activity."""
        summary = self.get_recent_social_proof_summary(seconds)
        
        score = 0.0
        
        # Base score from proof weight
        score += min(summary['total_proof_weight'] / 10.0, 1.0) * 0.4
        
        # User diversity bonus
        if summary['unique_users'] >= 3:
            score += 0.2
        elif summary['unique_users'] >= 5:
            score += 0.3
        
        # Reaction type bonuses
        if summary['reaction_types']['clip_request'] > 0:
            score += 0.3  # Clip requests are strong indicators
        if summary['reaction_types']['shock'] >= 2:
            score += 0.2
        if summary['reaction_types']['laughter'] >= 3:
            score += 0.2
        
        # Special user participation
        important_users = ['moderator', 'vip', 'subscriber', 'community_leader']
        important_participation = sum(
            summary['user_types'][user_type] for user_type in important_users
        )
        if important_participation >= 2:
            score += 0.2
        
        # Special events bonus
        if summary['special_events'] >= 1:
            score += 0.1
        
        # Viral indicators
        clip_worthy_count = summary['viral_indicators']['clip_worthy']
        if clip_worthy_count >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def add_message(self, message_data):
        """Add a message to the analysis history."""
        self.message_history.append(message_data)
        
        # Keep only recent messages (last 5 minutes)
        cutoff = time.time() - 300
        self.message_history = [
            msg for msg in self.message_history 
            if msg['timestamp'] >= cutoff
        ]
    
    def get_user_influence_ranking(self, limit=10):
        """Get top influential users in current session."""
        user_scores = []
        
        for username, profile in self.user_database.items():
            influence = self._calculate_influence_score(username)
            user_scores.append((username, influence, self._classify_user(username)))
        
        user_scores.sort(key=lambda x: x[1], reverse=True)
        return user_scores[:limit]
    
    def clear_old_data(self, hours=24):
        """Clear data older than specified hours."""
        cutoff = time.time() - (hours * 3600)
        
        # Clear old messages
        self.message_history = [
            msg for msg in self.message_history 
            if msg['timestamp'] >= cutoff
        ]
        
        # Clear old social signals
        for signal_type in self.social_signals:
            self.social_signals[signal_type] = [
                event for event in self.social_signals[signal_type]
                if event['timestamp'] >= cutoff
            ]