#!/usr/bin/env python3
"""
Real-Time Analytics Dashboard UI
Comprehensive dashboard showing viral detection performance, learning insights, and optimization tools
"""

import time
import json
from datetime import datetime, timedelta
from collections import Counter

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
                            QFrame, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
                            QTextEdit, QComboBox, QSlider, QSpinBox, QCheckBox, QGroupBox,
                            QScrollArea, QGridLayout, QSplitter, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush
import numpy as np


class AnalyticsDashboard(QWidget):
    """Comprehensive analytics dashboard for viral detection system."""
    
    # Signals
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, analytics_system, learning_system, config_manager):
        super().__init__()
        
        self.analytics_system = analytics_system
        self.learning_system = learning_system
        self.config_manager = config_manager
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dashboard UI."""
        self.setWindowTitle("🔥 Ultimate Viral Detection - Analytics Dashboard")
        self.setMinimumSize(1200, 800)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("🔥 ULTIMATE VIRAL DETECTION ANALYTICS")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ff6b35;
            padding: 10px;
            background-color: #2d2d2d;
            border-radius: 5px;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.setup_overview_tab()
        self.setup_performance_tab()
        self.setup_learning_tab()
        self.setup_optimization_tab()
        self.setup_settings_tab()
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_overview_tab(self):
        """Setup the overview tab with key metrics."""
        overview_widget = QWidget()
        layout = QVBoxLayout(overview_widget)
        
        # Real-time metrics grid
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        metrics_layout = QGridLayout(metrics_frame)
        
        # Create metric cards
        self.detection_rate_card = self.create_metric_card("Detection Rate", "0.0/min", "#4CAF50")
        self.success_rate_card = self.create_metric_card("Success Rate", "N/A", "#2196F3")
        self.confidence_card = self.create_metric_card("Avg Confidence", "0%", "#FF9800")
        self.clips_created_card = self.create_metric_card("Clips Created", "0", "#9C27B0")
        
        metrics_layout.addWidget(self.detection_rate_card, 0, 0)
        metrics_layout.addWidget(self.success_rate_card, 0, 1)
        metrics_layout.addWidget(self.confidence_card, 1, 0)
        metrics_layout.addWidget(self.clips_created_card, 1, 1)
        
        layout.addWidget(metrics_frame)
        
        # Signal quality indicators
        signals_frame = QFrame()
        signals_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        signals_layout = QVBoxLayout(signals_frame)
        
        signals_title = QLabel("📡 Signal Quality")
        signals_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        signals_layout.addWidget(signals_title)
        
        # Signal bars
        self.chat_signal_bar = self.create_signal_bar("💬 Chat Analysis", "#4CAF50")
        self.video_signal_bar = self.create_signal_bar("🎥 Video Analysis", "#2196F3")
        self.social_signal_bar = self.create_signal_bar("👥 Social Proof", "#FF9800")
        self.sync_signal_bar = self.create_signal_bar("⚡ Signal Sync", "#9C27B0")
        
        signals_layout.addWidget(self.chat_signal_bar)
        signals_layout.addWidget(self.video_signal_bar)
        signals_layout.addWidget(self.social_signal_bar)
        signals_layout.addWidget(self.sync_signal_bar)
        
        layout.addWidget(signals_frame)
        
        # Recent detections
        recent_frame = QFrame()
        recent_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        recent_layout = QVBoxLayout(recent_frame)
        
        recent_title = QLabel("🎯 Recent Viral Detections")
        recent_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        recent_layout.addWidget(recent_title)
        
        self.recent_detections_list = QListWidget()
        self.recent_detections_list.setMaximumHeight(200)
        self.recent_detections_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
        """)
        recent_layout.addWidget(self.recent_detections_list)
        
        layout.addWidget(recent_frame)
        
        self.tab_widget.addTab(overview_widget, "📊 Overview")
    
    def setup_performance_tab(self):
        """Setup the performance analysis tab."""
        performance_widget = QWidget()
        layout = QVBoxLayout(performance_widget)
        
        # Performance over time chart (placeholder)
        chart_frame = QFrame()
        chart_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        chart_layout = QVBoxLayout(chart_frame)
        
        chart_title = QLabel("📈 Performance Over Time")
        chart_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        chart_layout.addWidget(chart_title)
        
        # Simple performance visualization
        self.performance_chart = PerformanceChart()
        chart_layout.addWidget(self.performance_chart)
        
        layout.addWidget(chart_frame)
        
        # Context performance table
        context_frame = QFrame()
        context_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        context_layout = QVBoxLayout(context_frame)
        
        context_title = QLabel("🎮 Context Performance")
        context_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        context_layout.addWidget(context_title)
        
        self.context_table = QTableWidget()
        self.context_table.setColumnCount(4)
        self.context_table.setHorizontalHeaderLabels(["Context", "Detections", "Avg Confidence", "Success Rate"])
        self.context_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                color: white;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #404040;
                color: white;
                padding: 8px;
                border: none;
            }
        """)
        context_layout.addWidget(self.context_table)
        
        layout.addWidget(context_frame)
        
        self.tab_widget.addTab(performance_widget, "📈 Performance")
    
    def setup_learning_tab(self):
        """Setup the AI learning insights tab."""
        learning_widget = QWidget()
        layout = QVBoxLayout(learning_widget)
        
        # Learning insights
        insights_frame = QFrame()
        insights_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        insights_layout = QVBoxLayout(insights_frame)
        
        insights_title = QLabel("🧠 AI Learning Insights")
        insights_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        insights_layout.addWidget(insights_title)
        
        self.insights_text = QTextEdit()
        self.insights_text.setMaximumHeight(150)
        self.insights_text.setReadOnly(True)
        self.insights_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #4CAF50;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 8px;
                font-family: 'Courier New';
            }
        """)
        insights_layout.addWidget(self.insights_text)
        
        layout.addWidget(insights_frame)
        
        # Feature weights
        weights_frame = QFrame()
        weights_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        weights_layout = QVBoxLayout(weights_frame)
        
        weights_title = QLabel("⚖️ Learned Feature Weights")
        weights_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        weights_layout.addWidget(weights_title)
        
        # Weight bars
        self.chat_weight_bar = self.create_weight_bar("Chat Analysis")
        self.video_weight_bar = self.create_weight_bar("Video Analysis")
        self.social_weight_bar = self.create_weight_bar("Social Proof")
        self.momentum_weight_bar = self.create_weight_bar("Momentum")
        
        weights_layout.addWidget(self.chat_weight_bar)
        weights_layout.addWidget(self.video_weight_bar)
        weights_layout.addWidget(self.social_weight_bar)
        weights_layout.addWidget(self.momentum_weight_bar)
        
        layout.addWidget(weights_frame)
        
        # Streamer profile
        profile_frame = QFrame()
        profile_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        profile_layout = QVBoxLayout(profile_frame)
        
        profile_title = QLabel("👤 Streamer Profile")
        profile_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        profile_layout.addWidget(profile_title)
        
        self.profile_text = QTextEdit()
        self.profile_text.setMaximumHeight(150)
        self.profile_text.setReadOnly(True)
        self.profile_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #2196F3;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 8px;
            }
        """)
        profile_layout.addWidget(self.profile_text)
        
        layout.addWidget(profile_frame)
        
        self.tab_widget.addTab(learning_widget, "🧠 AI Learning")
    
    def setup_optimization_tab(self):
        """Setup the optimization recommendations tab."""
        optimization_widget = QWidget()
        layout = QVBoxLayout(optimization_widget)
        
        # Recommendations
        recommendations_frame = QFrame()
        recommendations_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        recommendations_layout = QVBoxLayout(recommendations_frame)
        
        recommendations_title = QLabel("💡 Optimization Recommendations")
        recommendations_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        recommendations_layout.addWidget(recommendations_title)
        
        self.recommendations_list = QListWidget()
        self.recommendations_list.setStyleSheet("""
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
                color: #ffd23f;
            }
        """)
        recommendations_layout.addWidget(self.recommendations_list)
        
        layout.addWidget(recommendations_frame)
        
        # Quick optimization actions
        actions_frame = QFrame()
        actions_frame.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        actions_layout = QVBoxLayout(actions_frame)
        
        actions_title = QLabel("⚡ Quick Actions")
        actions_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        actions_layout.addWidget(actions_title)
        
        buttons_layout = QHBoxLayout()
        
        auto_optimize_btn = QPushButton("🎯 Auto-Optimize Settings")
        auto_optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        auto_optimize_btn.clicked.connect(self.auto_optimize)
        buttons_layout.addWidget(auto_optimize_btn)
        
        reset_learning_btn = QPushButton("🔄 Reset Learning")
        reset_learning_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        reset_learning_btn.clicked.connect(self.reset_learning)
        buttons_layout.addWidget(reset_learning_btn)
        
        export_data_btn = QPushButton("📊 Export Analytics")
        export_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        export_data_btn.clicked.connect(self.export_analytics)
        buttons_layout.addWidget(export_data_btn)
        
        actions_layout.addLayout(buttons_layout)
        layout.addWidget(actions_frame)
        
        self.tab_widget.addTab(optimization_widget, "💡 Optimization")
    
    def setup_settings_tab(self):
        """Setup advanced settings tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Detection settings
        detection_group = QGroupBox("🎯 Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        
        # Sensitivity override
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Global Sensitivity:"))
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(10, 90)
        self.sensitivity_slider.setValue(70)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        self.sensitivity_label = QLabel("70%")
        sensitivity_layout.addWidget(self.sensitivity_label)
        
        detection_layout.addLayout(sensitivity_layout)
        
        # Detection mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detection Mode:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Conservative", "Balanced", "Aggressive", "Discovery"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        detection_layout.addLayout(mode_layout)
        
        layout.addWidget(detection_group)
        
        # Advanced settings
        advanced_group = QGroupBox("⚙️ Advanced Settings")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Auto-clipping
        self.auto_clip_check = QCheckBox("Auto-clip high confidence moments (>85%)")
        self.auto_clip_check.stateChanged.connect(self.on_auto_clip_changed)
        advanced_layout.addWidget(self.auto_clip_check)
        
        # Learning rate
        learning_layout = QHBoxLayout()
        learning_layout.addWidget(QLabel("AI Learning Rate:"))
        
        self.learning_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.learning_rate_slider.setRange(1, 20)
        self.learning_rate_slider.setValue(10)
        self.learning_rate_slider.valueChanged.connect(self.on_learning_rate_changed)
        learning_layout.addWidget(self.learning_rate_slider)
        
        self.learning_rate_label = QLabel("Normal")
        learning_layout.addWidget(self.learning_rate_label)
        
        advanced_layout.addLayout(learning_layout)
        
        # External validation
        self.external_validation_check = QCheckBox("Enable external viral validation")
        advanced_layout.addWidget(self.external_validation_check)
        
        layout.addWidget(advanced_group)
        
        # Save/Load settings
        settings_buttons_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)
        settings_buttons_layout.addWidget(save_btn)
        
        load_btn = QPushButton("📂 Load Settings")
        load_btn.clicked.connect(self.load_settings)
        settings_buttons_layout.addWidget(load_btn)
        
        layout.addLayout(settings_buttons_layout)
        layout.addStretch()
        
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")
    
    def create_metric_card(self, title, value, color):
        """Create a metric display card."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color}20;
                border: 2px solid {color};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)
        
        card.value_label = value_label  # Store reference for updates
        return card
    
    def create_signal_bar(self, name, color):
        """Create a signal quality bar."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        label = QLabel(name)
        label.setMinimumWidth(150)
        layout.addWidget(label)
        
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
                background-color: #333;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress)
        
        frame.progress = progress
        return frame
    
    def create_weight_bar(self, name):
        """Create a weight display bar."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        
        label = QLabel(name)
        label.setMinimumWidth(120)
        layout.addWidget(label)
        
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
                background-color: #333;
            }
            QProgressBar::chunk {
                background-color: #ff6b35;
                border-radius: 3px;
            }
        """)
        layout.addWidget(progress)
        
        weight_label = QLabel("0%")
        weight_label.setMinimumWidth(40)
        layout.addWidget(weight_label)
        
        frame.progress = progress
        frame.weight_label = weight_label
        return frame
    
    def update_dashboard(self):
        """Update all dashboard components."""
        try:
            # Update overview metrics
            self.update_overview_metrics()
            
            # Update performance data
            self.update_performance_data()
            
            # Update learning insights
            self.update_learning_data()
            
            # Update optimization recommendations
            self.update_optimization_data()
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
    
    def update_overview_metrics(self):
        """Update overview tab metrics."""
        summary = self.analytics_system.get_performance_summary('session')
        
        # Update metric cards
        detection_rate = summary.get('detection_rate_per_hour', 0)
        self.detection_rate_card.value_label.setText(f"{detection_rate:.1f}/hr")
        
        success_rate = summary.get('user_satisfaction_rate', 0)
        if success_rate > 0:
            self.success_rate_card.value_label.setText(f"{int(success_rate*100)}%")
        
        avg_confidence = summary.get('real_time_metrics', {}).get('average_confidence', 0)
        self.confidence_card.value_label.setText(f"{int(avg_confidence*100)}%")
        
        clips_created = summary.get('clips_created', 0)
        self.clips_created_card.value_label.setText(str(clips_created))
        
        # Update signal quality bars
        signal_quality = summary.get('real_time_metrics', {}).get('signal_quality', {})
        
        self.chat_signal_bar.progress.setValue(int(signal_quality.get('chat', 0) * 100))
        self.video_signal_bar.progress.setValue(int(signal_quality.get('video', 0) * 100))
        self.social_signal_bar.progress.setValue(int(signal_quality.get('social', 0) * 100))
        self.sync_signal_bar.progress.setValue(int(signal_quality.get('sync', 0) * 100))
        
        # Update recent detections
        self.update_recent_detections()
    
    def update_recent_detections(self):
        """Update recent detections list."""
        # Get recent detections from analytics system
        recent_detections = list(self.analytics_system.detection_results)[-10:]  # Last 10
        
        self.recent_detections_list.clear()
        
        for detection in recent_detections:
            time_ago = time.time() - detection.timestamp
            minutes_ago = int(time_ago // 60)
            
            confidence_color = "#4CAF50" if detection.confidence > 0.8 else "#ffd23f" if detection.confidence > 0.6 else "#ff6b35"
            
            item_text = (
                f"🎯 {minutes_ago}m ago - {int(detection.confidence*100)}% - "
                f"{detection.context.upper()}"
            )
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, detection)
            self.recent_detections_list.addItem(item)
    
    def update_performance_data(self):
        """Update performance tab data."""
        # Update context performance table
        summary = self.analytics_system.get_performance_summary('session')
        context_performance = summary.get('context_performance', {})
        
        self.context_table.setRowCount(len(context_performance))
        
        for row, (context, data) in enumerate(context_performance.items()):
            self.context_table.setItem(row, 0, QTableWidgetItem(context.title()))
            self.context_table.setItem(row, 1, QTableWidgetItem(str(data['detections'])))
            self.context_table.setItem(row, 2, QTableWidgetItem(f"{data['avg_confidence']*100:.1f}%"))
            self.context_table.setItem(row, 3, QTableWidgetItem("N/A"))  # Would need user feedback data
    
    def update_learning_data(self):
        """Update learning tab data."""
        insights = self.learning_system.get_learning_insights()
        
        # Update insights text
        insights_text = "🧠 AI LEARNING INSIGHTS:\n\n"
        for insight in insights:
            insights_text += f"• {insight}\n"
        
        self.insights_text.setText(insights_text)
        
        # Update feature weights
        weights = self.learning_system.feature_weights
        
        self.chat_weight_bar.progress.setValue(int(weights.get('chat_score', 0) * 100))
        self.chat_weight_bar.weight_label.setText(f"{weights.get('chat_score', 0)*100:.1f}%")
        
        self.video_weight_bar.progress.setValue(int(weights.get('video_score', 0) * 100))
        self.video_weight_bar.weight_label.setText(f"{weights.get('video_score', 0)*100:.1f}%")
        
        self.social_weight_bar.progress.setValue(int(weights.get('social_proof', 0) * 100))
        self.social_weight_bar.weight_label.setText(f"{weights.get('social_proof', 0)*100:.1f}%")
        
        self.momentum_weight_bar.progress.setValue(int(weights.get('momentum', 0) * 100))
        self.momentum_weight_bar.weight_label.setText(f"{weights.get('momentum', 0)*100:.1f}%")
        
        # Update streamer profile
        profile = self.learning_system.streamer_profile
        
        profile_text = "👤 STREAMER PROFILE:\n\n"
        profile_text += f"• Viral Patterns Learned: {len(profile.get('typical_viral_patterns', []))}\n"
        profile_text += f"• Best Detection Times: {len(profile.get('best_detection_times', []))}\n"
        profile_text += f"• Content Style: {profile.get('content_style', 'Unknown')}\n"
        
        if profile.get('best_detection_times'):
            hours = Counter(profile['best_detection_times']).most_common(3)
            profile_text += f"• Peak Hours: {', '.join([f'{h}:00' for h, _ in hours])}\n"
        
        self.profile_text.setText(profile_text)
    
    def update_optimization_data(self):
        """Update optimization tab data."""
        recommendations = self.analytics_system.get_optimization_recommendations()
        
        self.recommendations_list.clear()
        
        for rec in recommendations:
            item = QListWidgetItem(f"💡 {rec}")
            self.recommendations_list.addItem(item)
    
    # Settings event handlers
    def on_sensitivity_changed(self, value):
        """Handle sensitivity slider change."""
        self.sensitivity_label.setText(f"{value}%")
        self.settings_changed.emit({'sensitivity': value / 100.0})
    
    def on_mode_changed(self, mode):
        """Handle detection mode change."""
        self.settings_changed.emit({'mode': mode.lower()})
    
    def on_auto_clip_changed(self, state):
        """Handle auto-clip checkbox change."""
        self.settings_changed.emit({'auto_clip_high_confidence': state == Qt.CheckState.Checked})
    
    def on_learning_rate_changed(self, value):
        """Handle learning rate change."""
        rates = {1: "Very Slow", 5: "Slow", 10: "Normal", 15: "Fast", 20: "Very Fast"}
        rate_text = rates.get(value, "Normal")
        self.learning_rate_label.setText(rate_text)
        
        # Update learning system
        self.learning_system.learning_rate = value / 100.0
    
    # Action buttons
    def auto_optimize(self):
        """Auto-optimize settings based on current performance."""
        # This would implement automatic optimization logic
        print("Auto-optimizing settings...")
    
    def reset_learning(self):
        """Reset AI learning data."""
        # This would reset the learning system
        print("Resetting AI learning...")
    
    def export_analytics(self):
        """Export analytics data."""
        # This would export analytics to a file
        print("Exporting analytics...")
    
    def save_settings(self):
        """Save current settings."""
        settings = {
            'sensitivity': self.sensitivity_slider.value() / 100.0,
            'mode': self.mode_combo.currentText().lower(),
            'auto_clip': self.auto_clip_check.isChecked(),
            'learning_rate': self.learning_rate_slider.value() / 100.0,
            'external_validation': self.external_validation_check.isChecked()
        }
        
        # Save to config
        for key, value in settings.items():
            self.config_manager.set(f"viral_detection.{key}", value)
        
        print("Settings saved!")
    
    def load_settings(self):
        """Load settings from config."""
        sensitivity = self.config_manager.get("viral_detection.sensitivity", 0.7)
        self.sensitivity_slider.setValue(int(sensitivity * 100))
        
        mode = self.config_manager.get("viral_detection.mode", "balanced")
        self.mode_combo.setCurrentText(mode.title())
        
        auto_clip = self.config_manager.get("viral_detection.auto_clip", False)
        self.auto_clip_check.setChecked(auto_clip)
        
        print("Settings loaded!")
    
    def apply_dark_theme(self):
        """Apply dark theme to the dashboard."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-radius: 3px 3px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #ff6b35;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)


class PerformanceChart(QWidget):
    """Simple performance chart widget."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.data_points = []
    
    def add_data_point(self, value):
        """Add a new data point."""
        self.data_points.append(value)
        if len(self.data_points) > 50:  # Keep last 50 points
            self.data_points.pop(0)
        self.update()
    
    def paintEvent(self, event):
        """Paint the performance chart."""
        if not self.data_points:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor("#2d2d2d"))
        
        # Chart area
        margin = 20
        chart_rect = self.rect().adjusted(margin, margin, -margin, -margin)
        
        # Draw grid
        painter.setPen(QPen(QColor("#444"), 1))
        for i in range(1, 5):
            y = chart_rect.top() + (chart_rect.height() * i / 5)
            painter.drawLine(chart_rect.left(), y, chart_rect.right(), y)
        
        # Draw data
        if len(self.data_points) > 1:
            painter.setPen(QPen(QColor("#4CAF50"), 2))
            
            for i in range(1, len(self.data_points)):
                x1 = chart_rect.left() + ((i-1) * chart_rect.width() / (len(self.data_points)-1))
                y1 = chart_rect.bottom() - (self.data_points[i-1] * chart_rect.height())
                x2 = chart_rect.left() + (i * chart_rect.width() / (len(self.data_points)-1))
                y2 = chart_rect.bottom() - (self.data_points[i] * chart_rect.height())
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Title
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(chart_rect.top() + 10, 35, "Detection Success Rate Over Time")