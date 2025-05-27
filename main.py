#!/usr/bin/env python3
"""
BeastClipper v3.0 Ultimate Edition - Main Application (FIXED)
A comprehensive streaming clip automation tool with multi-platform export
"""

import sys
import os
import time
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import Counter

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QProgressBar,
                            QSlider, QCheckBox, QFrame, QTabWidget, QSpinBox, QTextEdit,
                            QMessageBox, QListWidget, QStatusBar, QFormLayout, QListWidgetItem,
                            QDialog, QDialogButtonBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# Import modules
from config import ConfigManager, TempFileManager, QTextEditLogger, DEFAULT_CLIPS_DIR
from stream import StreamBuffer, StreamMonitor, ClipCreator, ClipEditor  
from analysis import ContentAnalyzer, ChatMonitor, StreamingAnalyzer
from export import ExportPresets, ClipExporter
from detection import ViralMomentDetector, ViralMomentManager
from credentials import CredentialsManager

# Ultimate Viral Detection System imports
from ultimate_viral_system import UltimateViralDetector, ViralMoment
from social_proof_analyzer import SocialProofAnalyzer
from context_timing_system import ContextDetector, PerfectTimingOptimizer, MomentumTracker
from analytics_learning_system import AdvancedAnalytics, LearningSystem, ExternalValidationSystem, DetectionResult

# New enhanced viral detection system imports
from enhanced_viral_integration import ImprovedViralDetectionSystem, EnhancedViralMoment

# Configure logger
logger = logging.getLogger("BeastClipper")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Constants
APP_NAME = "BeastClipper"
APP_VERSION = "3.0 Ultimate"

# Define a unified temp directory in the project root
BASE_TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(BASE_TEMP_DIR, exist_ok=True)


class BeastClipperApp(QMainWindow):
    """Main application window."""
    
    # Custom signals
    log_message = pyqtSignal(str, str)  # level, message
    
    def __init__(self):
        super().__init__()
        
        # Initialize configuration
        self.config_manager = ConfigManager()
        
        # Initialize credentials manager
        self.credentials_manager = CredentialsManager(self.config_manager)
        
        # Initialize managers and monitors
        self.temp_manager = TempFileManager(self.config_manager)
        self.stream_buffer = None
        self.stream_monitor = StreamMonitor(self.config_manager)
        self.chat_monitor = None
        self.content_analyzer = None
        self.clip_exporter = None
        
        # Initialize ultimate viral detection system
        self.ultimate_detector = None
        self.enhanced_viral_detector = None
        self.social_analyzer = SocialProofAnalyzer()
        self.context_detector = ContextDetector()
        self.timing_optimizer = PerfectTimingOptimizer()
        self.momentum_tracker = MomentumTracker()
        self.analytics_system = AdvancedAnalytics(self.config_manager)
        self.learning_system = LearningSystem(self.config_manager)
        self.external_validator = ExternalValidationSystem()
        
        # Sensitivity debouncing timer
        self.sensitivity_timer = QTimer()
        self.sensitivity_timer.setSingleShot(True)
        self.sensitivity_timer.timeout.connect(self._apply_sensitivity_change)
        
        # Enhanced moment manager with analytics
        self.moment_manager = EnhancedViralMomentManager(max_moments=20)
        
        # Real-time analytics data
        self.analytics_data = {
            'detection_rate': 0.0,
            'success_rate': 0.0,
            'context_performance': {},
            'signal_quality': {},
            'learning_insights': [],
            'context_distribution': Counter()
        }
        
        # Connect log signal
        self.log_message.connect(self._handle_log_message)
        
        # Setup UI
        self.setup_ui()
        
        # Load configuration to UI
        self.load_config_to_ui()
        
        # List of clips
        self.clips = []
        self.selected_clip = None
        
        # Connect stream monitor signals
        self.stream_monitor.stream_live.connect(self.on_stream_live)
        self.stream_monitor.stream_offline.connect(self.on_stream_offline)
        self.stream_monitor.status_update.connect(self.update_status)
        
        # Only start monitor if auto-monitoring is enabled
        if self.config_manager.get("auto_monitor", False):
            self.log_info("Auto-monitoring enabled, starting stream monitor")
            self.stream_monitor.start()
        else:
            self.log_info("Auto-monitoring disabled, stream monitor will start when needed")
        
        # Setup timers
        self.setup_timers()
        
        # Load existing clips
        self.load_clips()
        
        # Check system requirements
        self.check_requirements()
        
        # Log startup
        self.log_info(f"{APP_NAME} v{APP_VERSION} started successfully")
    
    def setup_ui(self):
        """Set up the user interface with enhanced ad status."""
        # Window settings
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main tab widget
        self.tab_widget = QTabWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.clip_tab = QWidget()
        self.clips_tab = QWidget()
        self.export_tab = QWidget()
        self.settings_tab = QWidget()
        self.logs_tab = QWidget()
        self.vod_tab = QWidget()
        
        # Add tabs
        self.tab_widget.addTab(self.clip_tab, "📹 Clip Recorder")
        self.tab_widget.addTab(self.clips_tab, "🎬 Clips Library")
        self.tab_widget.addTab(self.export_tab, "📤 Export & Share")
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")
        self.tab_widget.addTab(self.logs_tab, "📝 Logs")
        self.tab_widget.addTab(self.vod_tab, "📺 VOD Analysis")
        
        # Setup individual tabs
        self.setup_clip_tab()
        self.setup_clips_tab()
        self.setup_export_tab()
        self.setup_settings_tab()
        self.setup_logs_tab()
        self.setup_enhanced_viral_moments_ui()
        self.setup_vod_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Ad status indicator to the UI
        self.ad_status_label = QLabel("🟢 Ready")
        self.ad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.ad_status_label)
        
        # Buffer status
        self.buffer_status_label = QLabel("Buffer: Not Active")
        self.status_bar.addPermanentWidget(self.buffer_status_label)
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_clip_tab(self):
        """Setup the clip recording tab."""
        layout = QVBoxLayout(self.clip_tab)
        
        # Stream URL input section
        stream_group = QFrame()
        stream_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; }")
        stream_layout = QVBoxLayout(stream_group)
        
        # URL input
        url_layout = QHBoxLayout()
        url_label = QLabel("Stream URL:")
        url_label.setMinimumWidth(100)
        url_layout.addWidget(url_label)
        
        self.stream_url_input = QLineEdit()
        self.stream_url_input.setPlaceholderText("Enter Twitch channel name (e.g., xqc) or URL")
        url_layout.addWidget(self.stream_url_input)
        
        stream_layout.addLayout(url_layout)
        
        # Stream settings
        settings_layout = QHBoxLayout()
        
        # Format selection
        format_label = QLabel("Format:")
        settings_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp4", "webm", "mkv"])
        self.format_combo.setMinimumWidth(80)
        settings_layout.addWidget(self.format_combo)
        
        settings_layout.addSpacing(20)
        
        # Resolution selection
        resolution_label = QLabel("Resolution:")
        settings_layout.addWidget(resolution_label)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1080p", "720p", "480p", "360p"])
        self.resolution_combo.setMinimumWidth(100)
        settings_layout.addWidget(self.resolution_combo)
        
        settings_layout.addSpacing(20)
        
        # Buffer duration
        buffer_label = QLabel("Buffer (seconds):")
        settings_layout.addWidget(buffer_label)
        
        self.buffer_duration_spin = QSpinBox()
        self.buffer_duration_spin.setRange(60, 600)
        self.buffer_duration_spin.setValue(300)
        self.buffer_duration_spin.setSuffix(" sec")
        settings_layout.addWidget(self.buffer_duration_spin)
        
        settings_layout.addStretch()
        stream_layout.addLayout(settings_layout)
        
        # Buffer control button
        self.buffer_button = QPushButton("Start Buffer")
        self.buffer_button.setMinimumHeight(40)
        self.buffer_button.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        self.buffer_button.clicked.connect(self.toggle_buffer)
        stream_layout.addWidget(self.buffer_button)
        
        # Buffer progress
        self.buffer_progress = QProgressBar()
        self.buffer_progress.setMinimumHeight(25)
        stream_layout.addWidget(self.buffer_progress)
        
        layout.addWidget(stream_group)
        
        # Clip creation section
        clip_group = QFrame()
        clip_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; }")
        clip_layout = QVBoxLayout(clip_group)
        
        # Time selection
        time_layout = QHBoxLayout()
        
        time_ago_label = QLabel("Start Time (seconds ago):")
        time_layout.addWidget(time_ago_label)
        
        self.time_ago_slider = QSlider(Qt.Horizontal)
        self.time_ago_slider.setRange(0, 300)
        self.time_ago_slider.setValue(30)
        self.time_ago_slider.valueChanged.connect(self.update_time_display)
        time_layout.addWidget(self.time_ago_slider)
        
        self.time_ago_label = QLabel("30s")
        self.time_ago_label.setMinimumWidth(50)
        time_layout.addWidget(self.time_ago_label)
        
        time_layout.addSpacing(20)
        
        duration_label = QLabel("Duration:")
        time_layout.addWidget(duration_label)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 60)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" sec")
        time_layout.addWidget(self.duration_spin)
        
        clip_layout.addLayout(time_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        
        output_label = QLabel("Output Directory:")
        output_label.setMinimumWidth(100)
        output_layout.addWidget(output_label)
        
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.config_manager.get("output_directory", DEFAULT_CLIPS_DIR))
        output_layout.addWidget(self.output_dir_input)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_button)
        
        clip_layout.addLayout(output_layout)
        
        # Create clip button
        self.create_clip_button = QPushButton("Create Clip")
        self.create_clip_button.setMinimumHeight(40)
        self.create_clip_button.setEnabled(False)
        self.create_clip_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        self.create_clip_button.clicked.connect(self.create_clip)
        clip_layout.addWidget(self.create_clip_button)
        
        # Clip progress
        self.clip_progress = QProgressBar()
        self.clip_progress.setMinimumHeight(25)
        clip_layout.addWidget(self.clip_progress)
        
        layout.addWidget(clip_group)
        layout.addStretch()
    
    def setup_clips_tab(self):
        """Setup the clips library tab."""
        layout = QVBoxLayout(self.clips_tab)
        
        # Clips list
        self.clips_list = QListWidget()
        self.clips_list.itemSelectionChanged.connect(self.on_clip_selected)
        layout.addWidget(self.clips_list)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_clip)
        controls_layout.addWidget(self.play_button)
        
        self.analyze_button = QPushButton("🔍 Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_clip)
        controls_layout.addWidget(self.analyze_button)
        
        self.export_button = QPushButton("📤 Export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.prepare_export)
        controls_layout.addWidget(self.export_button)
        
        self.delete_button = QPushButton("🗑 Delete")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_clip)
        controls_layout.addWidget(self.delete_button)
        
        layout.addLayout(controls_layout)
    
    def setup_export_tab(self):
        """Setup the export tab with improved layout and spacing."""
        layout = QVBoxLayout(self.export_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Export settings
        export_group = QFrame()
        export_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 15px; }")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)
        
        # Selected clip
        clip_layout = QHBoxLayout()
        clip_label = QLabel("Selected Clip:")
        clip_label.setMinimumWidth(120)
        clip_layout.addWidget(clip_label)
        
        self.selected_clip_label = QLineEdit()
        self.selected_clip_label.setReadOnly(True)
        self.selected_clip_label.setMinimumHeight(30)
        clip_layout.addWidget(self.selected_clip_label)
        export_layout.addLayout(clip_layout)
        
        # Platform selection
        platform_layout = QHBoxLayout()
        platform_label = QLabel("Export Platform:")
        platform_label.setMinimumWidth(120)
        platform_layout.addWidget(platform_label)
        
        self.platform_combo = QComboBox()
        self.platform_combo.setMinimumHeight(30)
        self.platform_combo.setMinimumWidth(200)
        self.platform_combo.addItems(["YouTube", "Twitch", "TikTok"])
        self.platform_combo.currentIndexChanged.connect(self.update_platform_info)
        platform_layout.addWidget(self.platform_combo)
        platform_layout.addStretch(1)
        export_layout.addLayout(platform_layout)
        
        # Platform info
        self.platform_info_label = QLabel()
        self.platform_info_label.setStyleSheet("background-color: #2d2d2d; padding: 10px; border-radius: 3px;")
        self.platform_info_label.setMinimumHeight(120)
        self.platform_info_label.setWordWrap(True)
        export_layout.addWidget(self.platform_info_label)
        
        # Caption/description field
        caption_label = QLabel("Caption/Description:")
        caption_label.setStyleSheet("margin-top: 10px;")
        export_layout.addWidget(caption_label)
        
        self.caption_input = QTextEdit()
        self.caption_input.setMinimumHeight(80)
        self.caption_input.setMaximumHeight(120)
        self.caption_input.setPlaceholderText("Enter a caption or description for your clip...")
        export_layout.addWidget(self.caption_input)
        
        # Export button
        self.export_button = QPushButton("Export Clip")
        self.export_button.setMinimumHeight(50)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 5px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        self.export_button.clicked.connect(self.export_for_platform)
        export_layout.addWidget(self.export_button)
        
        # Export progress
        self.export_progress = QProgressBar()
        self.export_progress.setMinimumHeight(25)
        export_layout.addWidget(self.export_progress)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        # Initial update
        self.update_platform_info()
    
    def setup_settings_tab(self):
        """Setup the settings tab."""
        layout = QVBoxLayout(self.settings_tab)
        
        # General settings
        general_group = QFrame()
        general_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; }")
        general_layout = QVBoxLayout(general_group)
        
        # Buffer settings
        buffer_layout = QHBoxLayout()
        buffer_label = QLabel("Default Buffer Duration:")
        buffer_label.setMinimumWidth(150)
        buffer_layout.addWidget(buffer_label)
        
        self.default_buffer_spin = QSpinBox()
        self.default_buffer_spin.setRange(60, 600)
        self.default_buffer_spin.setValue(self.config_manager.get("buffer_duration", 300))
        self.default_buffer_spin.setSuffix(" seconds")
        buffer_layout.addWidget(self.default_buffer_spin)
        general_layout.addLayout(buffer_layout)
        
        # Auto-export settings
        auto_export_layout = QHBoxLayout()
        self.auto_export_check = QCheckBox("Auto-export clips after creation")
        self.auto_export_check.setChecked(self.config_manager.get("auto_export", False))
        auto_export_layout.addWidget(self.auto_export_check)
        general_layout.addLayout(auto_export_layout)
        
        # Auto-monitor settings
        auto_monitor_layout = QHBoxLayout()
        self.auto_monitor_check = QCheckBox("Auto-monitor streams on startup (increases network usage)")
        self.auto_monitor_check.setChecked(self.config_manager.get("auto_monitor", False))
        self.auto_monitor_check.setToolTip("When disabled, stream monitoring only starts when you begin buffering")
        auto_monitor_layout.addWidget(self.auto_monitor_check)
        general_layout.addLayout(auto_monitor_layout)
        
        layout.addWidget(general_group)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        layout.addStretch()
    
    def setup_logs_tab(self):
        """Setup the logs tab."""
        layout = QVBoxLayout(self.logs_tab)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: Consolas, monospace; }")
        layout.addWidget(self.log_text)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        clear_button = QPushButton("Clear Logs")
        clear_button.clicked.connect(lambda: self.log_text.clear())
        controls_layout.addWidget(clear_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Setup log handler
        text_handler = QTextEditLogger(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)
    
    def setup_enhanced_viral_moments_ui(self):
        """Setup enhanced viral moments UI with analytics."""
        # Main viral detection group
        viral_group = QFrame()
        viral_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; margin-top: 10px; }")
        viral_layout = QVBoxLayout(viral_group)
        
        # Title row with status indicator
        title_row = QHBoxLayout()
        
        title_label = QLabel("🔥 IRC-POWERED VIRAL DETECTION")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #ff6b35;")
        title_row.addWidget(title_label)
        
        # Status indicator
        self.detection_status = QLabel("●")
        self.detection_status.setStyleSheet("color: #666; font-size: 20px;")
        title_row.addWidget(self.detection_status)
        
        self.detect_toggle = QPushButton("Enable IRC Detection")
        self.detect_toggle.setCheckable(True)
        self.detect_toggle.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #ff6b35;
                color: white;
            }
        """)
        self.detect_toggle.toggled.connect(self.toggle_enhanced_viral_detection)
        title_row.addWidget(self.detect_toggle)
        
        viral_layout.addLayout(title_row)
        
        # Real-time analytics row
        analytics_row = QHBoxLayout()
        
        self.detection_rate_label = QLabel("Rate: 0.0/hr")
        self.detection_rate_label.setStyleSheet("font-size: 11px; color: #888;")
        analytics_row.addWidget(self.detection_rate_label)
        
        self.success_rate_label = QLabel("Success: N/A")
        self.success_rate_label.setStyleSheet("font-size: 11px; color: #888;")
        analytics_row.addWidget(self.success_rate_label)
        
        self.context_label = QLabel("Context: Unknown")
        self.context_label.setStyleSheet("font-size: 11px; color: #888;")
        analytics_row.addWidget(self.context_label)
        
        analytics_row.addStretch()
        viral_layout.addLayout(analytics_row)
        
        # Detection mode selection
        mode_layout = QHBoxLayout()
        
        mode_label = QLabel("Mode:")
        mode_layout.addWidget(mode_label)
        
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(["Conservative", "Balanced", "Aggressive", "Discovery"])
        self.detection_mode_combo.setCurrentText("Balanced")
        self.detection_mode_combo.currentTextChanged.connect(self.change_detection_mode)
        mode_layout.addWidget(self.detection_mode_combo)
        
        mode_layout.addStretch()
        viral_layout.addLayout(mode_layout)
        
        # Enhanced sensitivity slider
        sensitivity_layout = QHBoxLayout()
        
        sensitivity_label = QLabel("AI Sensitivity:")
        sensitivity_layout.addWidget(sensitivity_label)
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 90)
        self.sensitivity_slider.setValue(70)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_slider_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        self.sensitivity_value = QLabel("70%")
        self.sensitivity_value.setMinimumWidth(40)
        sensitivity_layout.addWidget(self.sensitivity_value)
        
        viral_layout.addLayout(sensitivity_layout)
        
        # Signal quality indicators
        signals_layout = QHBoxLayout()
        
        self.chat_signal = QLabel("💬 IRC Chat: ●")
        self.chat_signal.setStyleSheet("font-size: 12px;")
        signals_layout.addWidget(self.chat_signal)
        
        self.video_signal = QLabel("🎥 Video: ●") 
        self.video_signal.setStyleSheet("font-size: 12px;")
        signals_layout.addWidget(self.video_signal)
        
        self.social_signal = QLabel("👥 Social: ●")
        self.social_signal.setStyleSheet("font-size: 12px;")
        signals_layout.addWidget(self.social_signal)
        
        # Add performance indicator
        self.performance_signal = QLabel("⚡ Fast: ●")
        self.performance_signal.setStyleSheet("font-size: 12px; color: #4CAF50;")
        signals_layout.addWidget(self.performance_signal)
        
        signals_layout.addStretch()
        viral_layout.addLayout(signals_layout)
        
        # Detected moments with enhanced display
        moments_label = QLabel("🎯 High-Confidence Viral Moments:")
        moments_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        viral_layout.addWidget(moments_label)
        
        self.moments_list = QListWidget()
        self.moments_list.setMaximumHeight(200)
        self.moments_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background-color: #ff6b35;
            }
        """)
        self.moments_list.itemDoubleClicked.connect(self.clip_viral_moment)
        viral_layout.addWidget(self.moments_list)
        
        # Enhanced action buttons
        actions_layout = QHBoxLayout()
        
        self.clip_moment_button = QPushButton("🎬 Create Viral Clip")
        self.clip_moment_button.setEnabled(False)
        self.clip_moment_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                padding: 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff8c5a;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        self.clip_moment_button.clicked.connect(self.clip_viral_moment)
        actions_layout.addWidget(self.clip_moment_button)
        
        self.feedback_button = QPushButton("📊 Analytics")
        self.feedback_button.clicked.connect(self.show_analytics_dashboard)
        actions_layout.addWidget(self.feedback_button)
        
        viral_layout.addLayout(actions_layout)
        
        # Add to main clip tab
        self.clip_tab.layout().addWidget(viral_group)

    def setup_vod_tab(self):
        """Setup the VOD analysis tab for online videos only."""
        layout = QVBoxLayout(self.vod_tab)
        url_group = QFrame()
        url_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; }")
        url_layout = QVBoxLayout(url_group)
        url_layout_row = QHBoxLayout()
        url_label = QLabel("Video URL:")
        url_label.setMinimumWidth(100)
        url_layout_row.addWidget(url_label)
        self.vod_url_input = QLineEdit()
        self.vod_url_input.setPlaceholderText("Enter YouTube URL or Twitch VOD URL")
        url_layout_row.addWidget(self.vod_url_input)
        url_layout.addLayout(url_layout_row)
        settings_layout = QHBoxLayout()
        sensitivity_label = QLabel("Detection Sensitivity:")
        settings_layout.addWidget(sensitivity_label)
        self.vod_sensitivity_slider = QSlider(Qt.Horizontal)
        self.vod_sensitivity_slider.setRange(1, 99)
        self.vod_sensitivity_slider.setValue(50)
        settings_layout.addWidget(self.vod_sensitivity_slider)
        self.vod_sensitivity_value = QLabel("50%")
        self.vod_sensitivity_slider.valueChanged.connect(lambda v: self.vod_sensitivity_value.setText(f"{v}%"))
        settings_layout.addWidget(self.vod_sensitivity_value)
        # Debug mode checkbox
        self.vod_debug_checkbox = QCheckBox("Enable Debug Visualization")
        settings_layout.addWidget(self.vod_debug_checkbox)
        url_layout.addLayout(settings_layout)
        self.analyze_vod_button = QPushButton("Analyze Video")
        self.analyze_vod_button.setMinimumHeight(40)
        self.analyze_vod_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        self.analyze_vod_button.clicked.connect(self.analyze_video_from_url)
        url_layout.addWidget(self.analyze_vod_button)
        progress_group = QVBoxLayout()
        analysis_label = QLabel("Analysis Progress:")
        progress_group.addWidget(analysis_label)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setMinimumHeight(25)
        progress_group.addWidget(self.analysis_progress)
        url_layout.addLayout(progress_group)
        layout.addWidget(url_group)
        results_group = QFrame()
        results_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 10px; }")
        results_layout = QVBoxLayout(results_group)
        results_label = QLabel("Detected Viral Moments:")
        results_layout.addWidget(results_label)
        self.vod_moments_list = QListWidget()
        results_layout.addWidget(self.vod_moments_list)
        self.download_clips_button = QPushButton("Download Selected Viral Clips")
        self.download_clips_button.setEnabled(False)
        self.download_clips_button.clicked.connect(self.download_viral_clips)
        results_layout.addWidget(self.download_clips_button)
        self.download_all_clips_button = QPushButton("Download All Viral Clips")
        self.download_all_clips_button.setEnabled(False)
        self.download_all_clips_button.clicked.connect(self.download_all_viral_clips)
        results_layout.addWidget(self.download_all_clips_button)
        download_label = QLabel("Download Progress:")
        results_layout.addWidget(download_label)
        self.download_progress = QProgressBar()
        self.download_progress.setMinimumHeight(25)
        results_layout.addWidget(self.download_progress)
        layout.addWidget(results_group)

    def setup_timers(self):
        """Setup application timers."""
        # Buffer status timer
        self.buffer_timer = QTimer()
        self.buffer_timer.timeout.connect(self.update_buffer_status)
        self.buffer_timer.start(1000)  # Update every second
        
        # Clips refresh timer
        self.clips_timer = QTimer()
        self.clips_timer.timeout.connect(self.load_clips)
        self.clips_timer.start(30000)  # Refresh every 30 seconds
        
        # Analytics update timer
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.update_analytics_display)
        self.analytics_timer.start(2000)  # Update every 2 seconds
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        dark_theme = """
            QMainWindow, QWidget {
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
            }
            QTabBar::tab:selected {
                background-color: #404040;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                color: #ffffff;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555;
                color: #ffffff;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                color: #ffffff;
            }
            QListWidget::item:selected {
                background-color: #404040;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #444;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """
        self.setStyleSheet(dark_theme)
    
    def load_config_to_ui(self):
        """Load configuration values to UI elements."""
        # Load default settings
        self.format_combo.setCurrentText(self.config_manager.get("format", "mp4"))
        self.resolution_combo.setCurrentText(self.config_manager.get("resolution", "1080p"))
        self.buffer_duration_spin.setValue(self.config_manager.get("buffer_duration", 300))
    
    def check_requirements(self):
        """Check if required tools are available."""
        required_tools = ["streamlink"]
        recommended_tools = ["ffmpeg", "ffprobe"]
        missing_required = []
        missing_recommended = []
        
        # Check required tools
        for tool in required_tools:
            try:
                result = subprocess.run([tool, "--version"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing_required.append(tool)
            except FileNotFoundError:
                missing_required.append(tool)
        
        # Check recommended tools
        for tool in recommended_tools:
            try:
                result = subprocess.run([tool, "-version"], capture_output=True, text=True)
                if result.returncode != 0:
                    missing_recommended.append(tool)
            except FileNotFoundError:
                missing_recommended.append(tool)
        
        if missing_required:
            self.log_error(f"Missing required tools: {', '.join(missing_required)}")
            QMessageBox.critical(
                self,
                "Missing Requirements",
                f"The following required tools are missing: {', '.join(missing_required)}\n"
                "BeastClipper cannot run without these tools.\n\n"
                "Install with: pip install streamlink"
            )
            sys.exit(1)
        
        if missing_recommended:
            self.log_warning(f"Missing recommended tools: {', '.join(missing_recommended)}")
            QMessageBox.warning(
                self,
                "Missing Recommended Tools",
                f"The following tools are missing: {', '.join(missing_recommended)}\n"
                "Clip creation may not work without FFmpeg.\n\n"
                "Install FFmpeg from: https://ffmpeg.org/download.html"
            )

    # === FIXED MISSING METHODS ===
    
    def start_viral_detection(self):
        """Start the IRC-powered enhanced viral detection system (no browser!)."""
        if not self.stream_buffer or not self.stream_buffer.isRunning():
            self.log_error("Cannot start viral detection without active buffer")
            return
            
        self.log_info("🚀 Starting IRC-Powered Enhanced Viral Detection System...")
        
        # Create enhanced detector with professional IRC
        if not self.enhanced_viral_detector:
            self.enhanced_viral_detector = ImprovedViralDetectionSystem(
                stream_url=self.stream_url_input.text(),
                stream_buffer=self.stream_buffer,
                config_manager=self.config_manager,
                sensitivity=self.sensitivity_slider.value() / 100.0
            )
            
            # Connect signals
            self.enhanced_viral_detector.viral_moment_detected.connect(self.on_enhanced_viral_moment)
            self.enhanced_viral_detector.confidence_update.connect(self.on_confidence_update)
            self.enhanced_viral_detector.analytics_update.connect(self.on_analytics_update)
            self.enhanced_viral_detector.status_update.connect(self.update_status)
            self.enhanced_viral_detector.error_occurred.connect(self.log_error)
            self.enhanced_viral_detector.debug_info.connect(self.on_debug_info)
        
        # Start detection
        self.enhanced_viral_detector.start()
        self.update_status("🔥 IRC-powered viral detection active! (No browser needed)")
    
    def stop_viral_detection(self):
        """Stop the IRC-powered enhanced viral detection system."""
        if self.enhanced_viral_detector and self.enhanced_viral_detector.isRunning():
            self.log_info("Stopping IRC-powered enhanced viral detection...")
            self.enhanced_viral_detector.stop()
            self.enhanced_viral_detector.wait()
            self.update_status("Viral detection stopped")
            
    def on_debug_info(self, debug_data):
        """Handle debug information from enhanced detector."""
        if self.config_manager.get("viral_detection.debug_mode", False):
            self.log_info(f"VIRAL DEBUG: {debug_data}")
            
    def on_enhanced_viral_moment(self, viral_moment: EnhancedViralMoment):
        """Handle detection of enhanced viral moment."""
        # Convert to compatible format for existing UI
        moment_data = {
            'timestamp': viral_moment.timestamp,
            'score': viral_moment.confidence,
            'confidence': viral_moment.confidence,
            'detected_at': viral_moment.detected_at,
            'time_ago': viral_moment.timestamp,
            'description': viral_moment.description,
            'context': viral_moment.context,
            'social_proof': {
                'unique_chatters': viral_moment.unique_chatters,
                'message_velocity': viral_moment.message_velocity,
                'emote_usage': viral_moment.emote_usage
            },
            'signals': {
                'chat': viral_moment.chat_score,
                'video': viral_moment.video_score,
                'audio': viral_moment.audio_score,
                'social': viral_moment.social_proof_score,
                'momentum': viral_moment.momentum_score
            },
            'uniqueness': viral_moment.uniqueness_score,
            'viral_potential': viral_moment.viral_potential,
            'clip_start': viral_moment.clip_start,
            'clip_end': viral_moment.clip_end,
            'timing_confidence': viral_moment.timing_confidence,
            'enhanced_data': viral_moment.to_dict()  # Full data for advanced features
        }
        
        self.moment_manager.add_moment(moment_data)
        
        # Update UI
        self.update_moments_list()
        
        # Enhanced auto-clip logic
        if (viral_moment.confidence >= 0.85 and 
            viral_moment.viral_potential >= 0.8 and
            self.config_manager.get("viral_detection.auto_clip_high_confidence", False)):
            
            self.log_info(f"Auto-clipping ENHANCED viral moment (conf:{viral_moment.confidence:.3f}, potential:{viral_moment.viral_potential:.3f})")
            # Create list item for the moment
            item_text = f"🔥 ENHANCED AUTO: {viral_moment.description} [{int(viral_moment.confidence*100)}%]"
            item = QListWidgetItem(item_text)
            self.clip_viral_moment(item)
    
    def clip_viral_moment(self, item=None):
        """Create a clip from a detected viral moment."""
        if not self.stream_buffer:
            self.show_error("No active buffer")
            return
        
        # Get selected moment
        if item is None:
            selected_items = self.moments_list.selectedItems()
            if not selected_items:
                self.show_error("Please select a viral moment to clip")
                return
            item = selected_items[0]
        
        # Get moment data
        moment_index = self.moments_list.row(item)
        moment = self.moment_manager.get_moment_by_index(moment_index)
        
        if not moment:
            self.show_error("Invalid moment selected")
            return
        
        # Extract timing info from moment data
        if isinstance(moment, dict):
            # Handle enhanced viral moment data
            clip_start = moment.get('clip_start', moment.get('timestamp', 30))
            clip_end = moment.get('clip_end', clip_start + 20)
            duration = clip_end - clip_start
        else:
            # Handle regular moment
            time_ago = moment.get('time_ago', 30)
            duration = 20  # Default 20 second clips
        
        # Create output filename with viral marker
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        confidence = moment.get('confidence', moment.get('score', 0.5))
        filename = f"VIRAL_{timestamp}_conf{int(confidence*100)}.mp4"
        output_path = os.path.join(self.output_dir_input.text(), filename)
        
        self.log_info(f"Creating viral clip: {filename}")
        
        # Create enhanced clip creator with metadata
        self.clip_creator = EnhancedClipCreator(
            stream_buffer=self.stream_buffer,
            start_time_ago=clip_start if isinstance(moment, dict) else time_ago,
            duration=duration,
            output_path=output_path,
            format_type="mp4",
            moment_metadata=moment
        )
        
        # Connect signals
        self.clip_creator.progress_update.connect(self.on_clip_progress)
        self.clip_creator.status_update.connect(self.update_status)
        self.clip_creator.clip_created.connect(self.on_viral_clip_created)
        self.clip_creator.error_occurred.connect(self.on_clip_error)
        
        # Start creation
        self.clip_creator.start()
        self.update_status("Creating viral clip...")
        
        # Record detection for analytics
        if hasattr(moment, '__dict__'):  # ViralMoment object
            detection_result = DetectionResult(
                timestamp=moment.timestamp,
                confidence=moment.confidence,
                context=moment.context,
                signals=moment.signals,
                social_proof=moment.social_proof,
                clip_start=moment.clip_start,
                clip_end=moment.clip_end,
                was_clipped=True,
                clip_file=output_path
            )
            self.analytics_system.record_detection(detection_result)
    
    def update_moments_list(self):
        """Update the viral moments list display with enhanced information."""
        self.moments_list.clear()
        
        moments = self.moment_manager.get_moments()
        
        for i, moment in enumerate(moments):
            confidence = moment.get('confidence', moment.get('score', 0))
            context = moment.get('context', 'unknown')
            description = moment.get('description', 'Viral moment detected')
            
            # Enhanced display with additional metrics
            viral_potential = moment.get('viral_potential', 0)
            unique_chatters = moment.get('social_proof', {}).get('unique_chatters', 0)
            
            # Format display with emojis and enhanced info
            if confidence >= 0.9 and viral_potential >= 0.8:
                emoji = "💥"  # Exceptional moment
            elif confidence >= 0.8:
                emoji = "🔥"
            elif confidence >= 0.6:
                emoji = "⚡"
            else:
                emoji = "✨"
            
            # Enhanced item text with more details
            item_text = f"{emoji} {description} [{int(confidence*100)}%]"
            if unique_chatters > 0:
                item_text += f" - {unique_chatters} users"
            if viral_potential > 0:
                item_text += f" - {int(viral_potential*100)}% viral potential"
            item_text += f" - {context}"
            
            item = QListWidgetItem(item_text)
            
            # Enhanced color coding
            if confidence >= 0.9 and viral_potential >= 0.8:
                item.setForeground(Qt.magenta)  # Exceptional
            elif confidence >= 0.8:
                item.setForeground(Qt.red)
            elif confidence >= 0.6:
                item.setForeground(Qt.yellow)
            
            self.moments_list.addItem(item)
        
        # Enable clip button if moments exist
        self.clip_moment_button.setEnabled(len(moments) > 0)
    
    def on_confidence_update(self, confidence_data):
        """Update confidence metrics display."""
        self.analytics_data.update(confidence_data)
    
    def on_analytics_update(self, analytics):
        """Handle analytics updates from detector."""
        self.analytics_data.update(analytics)
        
        # Update context distribution
        if 'current_context' in analytics:
            self.analytics_data['context_distribution'][analytics['current_context']] += 1
    
    def toggle_enhanced_viral_detection(self, enabled):
        """Toggle enhanced viral detection system."""
        if enabled:
            if not self.stream_buffer or not self.stream_buffer.isRunning():
                self.show_error("Please start buffer before enabling viral detection")
                self.detect_toggle.setChecked(False)
                return
                
            self.start_viral_detection()
            self.detection_status.setText("●")
            self.detection_status.setStyleSheet("color: #4CAF50; font-size: 20px;")
        else:
            self.stop_viral_detection()
            self.detection_status.setText("●") 
            self.detection_status.setStyleSheet("color: #666; font-size: 20px;")

    def update_enhanced_sensitivity(self, value):
        """Update sensitivity with enhanced learning system integration."""
        self.sensitivity_value.setText(f"{value}%")
        sensitivity = value / 100.0
        
        # Update enhanced detector if running
        if self.enhanced_viral_detector and self.enhanced_viral_detector.isRunning():
            self.enhanced_viral_detector.update_sensitivity(sensitivity)
        
        
        # Save to config
        self.config_manager.set("viral_detection.sensitivity", sensitivity)

    def change_detection_mode(self, mode):
        """Change detection mode."""
        mode_lower = mode.lower()
        if self.enhanced_viral_detector and self.enhanced_viral_detector.isRunning():
            self.enhanced_viral_detector.set_detection_mode(mode_lower)
        
        self.config_manager.set("viral_detection.mode", mode_lower)
        self.log_info(f"Detection mode changed to: {mode}")

    def update_analytics_display(self):
        """Update real-time analytics display with enhanced metrics."""
        analytics = self.analytics_data
        
        # Update rate displays
        detection_rate = analytics.get('detection_rate', 0.0)
        self.detection_rate_label.setText(f"Rate: {detection_rate:.1f}/hr")
        
        success_rate = analytics.get('success_rate', 0.0)
        if success_rate > 0:
            self.success_rate_label.setText(f"Success: {int(success_rate*100)}%")
            self.success_rate_label.setStyleSheet(f"font-size: 11px; color: {'#4CAF50' if success_rate > 0.7 else '#ffd23f' if success_rate > 0.4 else '#f44336'};")
        
        # Update context with enhanced info
        context_dist = analytics.get('context_distribution', {})
        current_context = analytics.get('current_context', 'unknown')
        if current_context != 'unknown':
            self.context_label.setText(f"Context: {current_context.title()}")
        
        # Update signal quality indicators with enhanced data
        signal_quality = analytics.get('signal_quality', {})
        
        chat_quality = signal_quality.get('chat', 0.0)
        self.chat_signal.setText(f"💬 IRC Chat: {'●●●' if chat_quality > 0.8 else '●●' if chat_quality > 0.5 else '●' if chat_quality > 0.2 else '○'}")
        self.chat_signal.setStyleSheet(f"font-size: 12px; color: {'#4CAF50' if chat_quality > 0.7 else '#ffd23f' if chat_quality > 0.4 else '#666'};")
        
        video_quality = signal_quality.get('video', 0.0)
        self.video_signal.setText(f"🎥 Video: {'●●●' if video_quality > 0.8 else '●●' if video_quality > 0.5 else '●' if video_quality > 0.2 else '○'}")
        self.video_signal.setStyleSheet(f"font-size: 12px; color: {'#4CAF50' if video_quality > 0.7 else '#ffd23f' if video_quality > 0.4 else '#666'};")
        
        social_quality = signal_quality.get('social', 0.0)
        self.social_signal.setText(f"👥 Social: {'●●●' if social_quality > 0.8 else '●●' if social_quality > 0.5 else '●' if social_quality > 0.2 else '○'}")
        self.social_signal.setStyleSheet(f"font-size: 12px; color: {'#4CAF50' if social_quality > 0.7 else '#ffd23f' if social_quality > 0.4 else '#666'};")
        
        # Show processing performance if available
        processing_time = analytics.get('processing_time', 0)
        if processing_time > 0:
            performance_color = '#4CAF50' if processing_time < 0.5 else '#ffd23f' if processing_time < 1.0 else '#f44336'
            self.performance_signal.setText(f"⚡ Fast: {'●●●' if processing_time < 0.3 else '●●' if processing_time < 0.7 else '●'}")
            self.performance_signal.setStyleSheet(f"font-size: 12px; color: {performance_color};")
        else:
            # Default to fast since IRC is fast
            self.performance_signal.setText("⚡ Fast: ●●●")
            self.performance_signal.setStyleSheet("font-size: 12px; color: #4CAF50;")

    def show_analytics_dashboard(self):
        """Show enhanced analytics dashboard with comprehensive metrics."""
        dialog = QDialog(self)
        dialog.setWindowTitle("🔥 IRC-Powered Enhanced Viral Detection Analytics")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Get enhanced analytics data
        summary = self.analytics_system.get_performance_summary('session')
        recommendations = self.analytics_system.get_optimization_recommendations()
        insights = self.learning_system.get_learning_insights()
        
        # Get performance metrics from enhanced detector
        if self.enhanced_viral_detector:
            performance_metrics = self.enhanced_viral_detector.get_performance_metrics()
        else:
            performance_metrics = {}
        
        # Create enhanced tabs
        tabs = QTabWidget()
        
        # Enhanced Performance tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout(perf_widget)
        
        perf_text = QTextEdit()
        perf_text.setReadOnly(True)
        perf_content = f"""
📊 IRC-POWERED SESSION PERFORMANCE:
• Total Detections: {summary.get('total_detections', 0)}
• Clips Created: {summary.get('clips_created', 0)}
• Detection Rate: {summary.get('detection_rate_per_hour', 0):.1f}/hour
• Success Rate: {summary.get('user_satisfaction_rate', 0)*100:.1f}%
• Avg Confidence: {summary.get('real_time_metrics', {}).get('average_confidence', 0)*100:.1f}%

⚡ SYSTEM PERFORMANCE (IRC-powered):
• Avg Processing Time: {performance_metrics.get('avg_processing_time', 0):.3f}s
• Signals Processed: {performance_metrics.get('signals_processed', 0)}
• Moments Detected: {performance_metrics.get('moments_detected', 0)}
• IRC Benefits: Faster, more reliable, no browser dependency

🎮 CONTEXT PERFORMANCE:
"""
        
        for ctx, perf in summary.get('context_performance', {}).items():
            perf_content += f"• {ctx}: {perf.get('avg_confidence', 0)*100:.0f}% avg, {perf.get('detections', 0)} detections\n"
        
        # Signal quality metrics
        signal_quality = performance_metrics.get('signal_quality', {})
        if signal_quality:
            perf_content += f"\n📡 SIGNAL QUALITY:\n"
            for signal_type, quality in signal_quality.items():
                perf_content += f"• {signal_type.title()}: {quality*100:.0f}%\n"
        
        perf_text.setPlainText(perf_content)
        perf_layout.addWidget(perf_text)
        tabs.addTab(perf_widget, "📊 Enhanced Performance")
        
        # AI Learning tab
        learn_widget = QWidget()
        learn_layout = QVBoxLayout(learn_widget)
        
        learn_text = QTextEdit()
        learn_text.setReadOnly(True)
        learn_content = "🧠 AI LEARNING INSIGHTS:\n\n"
        for insight in insights:
            learn_content += f"• {insight}\n"
        
        learn_text.setPlainText(learn_content)
        learn_layout.addWidget(learn_text)
        tabs.addTab(learn_widget, "🧠 AI Learning")
        
        # Recommendations tab
        rec_widget = QWidget()
        rec_layout = QVBoxLayout(rec_widget)
        
        rec_text = QTextEdit()
        rec_text.setReadOnly(True)
        rec_content = "💡 OPTIMIZATION RECOMMENDATIONS:\n\n"
        for rec in recommendations:
            rec_content += f"• {rec}\n"
        
        rec_text.setPlainText(rec_content)
        rec_layout.addWidget(rec_text)
        tabs.addTab(rec_widget, "💡 Recommendations")
        
        layout.addWidget(tabs)
        
        # Enhanced buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        
        # Add export button for enhanced data
        export_btn = button_box.addButton("Export Enhanced Data", QDialogButtonBox.ActionRole)
        export_btn.clicked.connect(lambda: self.export_enhanced_analytics())
        
        layout.addWidget(button_box)
        
        dialog.exec_()

    def export_enhanced_analytics(self):
        """Export enhanced analytics data to JSON file."""
        try:
            if self.enhanced_viral_detector:
                performance_data = self.enhanced_viral_detector.get_performance_metrics()
                
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'session_summary': self.analytics_system.get_performance_summary('session'),
                    'performance_metrics': performance_data,
                    'learning_insights': self.learning_system.get_learning_insights(),
                    'recent_moments': [moment.get('enhanced_data', {}) for moment in self.moment_manager.get_moments()]
                }
                
                filename = f"enhanced_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.config_manager.get("output_directory", "."), filename)
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.show_info(f"Enhanced analytics exported to:\n{filepath}", "Export Complete")
                
        except Exception as e:
            self.show_error(f"Failed to export analytics: {str(e)}")

    def on_viral_clip_created(self, file_path):
        """Handle viral clip creation completion."""
        self.on_clip_created(file_path)
        
        # Show notification
        QMessageBox.information(
            self,
            "Viral Clip Created!",
            f"🔥 Your viral clip has been created successfully!\n\n{os.path.basename(file_path)}"
        )

    # === END OF FIXED METHODS ===
    
    def toggle_buffer(self):
        """Start or stop the stream buffer with enhanced connections."""
        if self.stream_buffer and self.stream_buffer.isRunning():
            # Stop buffer (existing code remains the same)
            self.log_info("Stopping stream buffer...")
            
            if self.enhanced_viral_detector and self.enhanced_viral_detector.isRunning():
                self.stop_viral_detection()
                self.detect_toggle.setChecked(False)
                
            self.stream_buffer.stop()
            self.stream_buffer.wait()
            self.stream_buffer = None
            
            self.buffer_button.setText("Start Buffer")
            self.buffer_button.setStyleSheet(self.buffer_button.styleSheet().replace("#d32f2f", "#2e7d32"))
            self.buffer_progress.setValue(0)
            self.create_clip_button.setEnabled(False)
            self.time_ago_slider.setMaximum(0)
            
            # Reset ad status
            self.ad_status_label.setText("🟢 Ready")
            self.ad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            self.update_status("Buffer stopped")
        else:
            # Start buffer (enhanced version)
            stream_url = self.stream_url_input.text().strip()
            
            if not stream_url:
                self.show_error("Please enter a Twitch stream URL or channel name")
                return
            
            # Validate Twitch URL/channel
            if not any(x in stream_url.lower() for x in ["twitch.tv", "twitch"]):
                stream_url = f"https://twitch.tv/{stream_url}"
            
            if "twitch.tv" not in stream_url.lower():
                self.show_error("Only Twitch streams are supported. Enter a channel name or twitch.tv URL")
                return
            
            # Make sure stream monitor is running
            if not self.stream_monitor.isRunning():
                self.stream_monitor.start()
            
            self.log_info(f"Starting enhanced Twitch stream buffer for: {stream_url}")
            
            # Create enhanced buffer
            self.stream_buffer = StreamBuffer(
                stream_url=stream_url,
                buffer_duration=self.buffer_duration_spin.value(),
                resolution=self.resolution_combo.currentText(),
                temp_manager=self.temp_manager
            )
            
            # Connect all signals including the new ad status signal
            self.stream_buffer.buffer_progress.connect(self.on_buffer_progress)
            self.stream_buffer.status_update.connect(self.update_status)
            self.stream_buffer.error_occurred.connect(self.on_buffer_error)
            self.stream_buffer.stream_info_updated.connect(self.on_stream_info_updated)
            self.stream_buffer.ad_status_update.connect(self.on_ad_status_update)  # NEW
            
            # Start buffer
            self.stream_buffer.start()
            
            self.buffer_button.setText("Stop Buffer")
            self.buffer_button.setStyleSheet(self.buffer_button.styleSheet().replace("#2e7d32", "#d32f2f"))
            self.create_clip_button.setEnabled(True)
            
            # Restart viral detection if it was enabled
            if self.detect_toggle.isChecked():
                self.start_viral_detection()
                
            self.update_status("Starting enhanced buffer with ad handling...")
    
    def create_clip(self):
        """Create a clip from the buffer."""
        if not self.stream_buffer:
            self.show_error("No active buffer")
            return
        
        # Get clip parameters
        time_ago = self.time_ago_slider.value()
        duration = self.duration_spin.value()
        output_dir = self.output_dir_input.text()
        format_type = self.format_combo.currentText()
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{timestamp}.{format_type}"
        output_path = os.path.join(output_dir, filename)
        
        self.log_info(f"Creating clip: {filename}")
        
        # Create clip creator
        self.clip_creator = ClipCreator(
            stream_buffer=self.stream_buffer,
            start_time_ago=time_ago,
            duration=duration,
            output_path=output_path,
            format_type=format_type
        )
        
        # Connect signals
        self.clip_creator.progress_update.connect(self.on_clip_progress)
        self.clip_creator.status_update.connect(self.update_status)
        self.clip_creator.clip_created.connect(self.on_clip_created)
        self.clip_creator.error_occurred.connect(self.on_clip_error)
        
        # Start creation
        self.clip_creator.start()
        
        self.create_clip_button.setEnabled(False)
        self.update_status("Creating clip...")
    
    def load_clips(self):
        """Load clips from the output directory."""
        output_dir = self.output_dir_input.text()
        
        if not os.path.exists(output_dir):
            return
        
        self.clips = []
        
        # Find all video files
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        
        for filename in os.listdir(output_dir):
            if any(filename.endswith(ext) for ext in video_extensions):
                file_path = os.path.join(output_dir, filename)
                file_stats = os.stat(file_path)
                
                self.clips.append({
                    'name': filename,
                    'path': file_path,
                    'size': file_stats.st_size,
                    'created': file_stats.st_ctime
                })
        
        # Sort by creation time (newest first)
        self.clips.sort(key=lambda x: x['created'], reverse=True)
        
        # Update list widget
        self.clips_list.clear()
        
        for clip in self.clips:
            size_mb = clip['size'] / (1024 * 1024)
            created_time = datetime.fromtimestamp(clip['created']).strftime("%Y-%m-%d %H:%M")
            item_text = f"{clip['name']} ({size_mb:.1f} MB) - {created_time}"
            self.clips_list.addItem(item_text)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_input.text()
        )
        
        if directory:
            self.output_dir_input.setText(directory)
            self.config_manager.set("output_directory", directory)
            self.load_clips()
    
    def update_time_display(self, value):
        """Update time ago display."""
        self.time_ago_label.setText(f"{value}s")
    
    def update_buffer_status(self):
        """Enhanced buffer status update with method and ad info."""
        if self.stream_buffer and self.stream_buffer.isRunning():
            status = self.stream_buffer.get_buffer_status()
            duration = status['duration']
            max_duration = status['max_duration']
            method = status.get('method', 'unknown').replace('_method_', '').replace('_', ' ').title()
            ad_detected = status.get('ad_detected', False)
            
            # Build status text
            buffer_text = f"Buffer: {duration}s / {max_duration}s"
            if method != 'Unknown':
                buffer_text += f" ({method})"
            
            if ad_detected:
                buffer_text += " [Waiting for ads]"
            
            self.buffer_status_label.setText(buffer_text)
            self.time_ago_slider.setMaximum(duration)
            
            if duration > 0 and not self.create_clip_button.isEnabled():
                self.create_clip_button.setEnabled(True)
        else:
            self.buffer_status_label.setText("Buffer: Not Active")
            # Reset ad status when buffer is not active
            if hasattr(self, 'ad_status_label'):
                self.ad_status_label.setText("🟢 Ready")
                self.ad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    def update_platform_info(self):
        """Update platform info based on selection with improved formatting."""
        if not hasattr(self, 'platform_combo') or not hasattr(self, 'platform_info_label'):
            return
            
        platform = self.platform_combo.currentText().lower()
        preset = ExportPresets.get_preset(platform)
        
        info_text = (
            f"<h3 style='margin-top: 0;'>{platform.capitalize()} Export Settings:</h3>"
            f"<ul style='margin-left: 15px; margin-top: 5px;'>"
            f"<li><b>Format:</b> {preset['format']}</li>"
            f"<li><b>Resolution:</b> {preset['resolution']}</li>"
            f"<li><b>FPS:</b> {preset['fps']}</li>"
            f"<li><b>Video Bitrate:</b> {preset['bitrate']}</li>"
            f"<li><b>Audio Bitrate:</b> {preset['audio_bitrate']}</li>"
        )
        
        if "max_duration" in preset:
            info_text += f"<li><b>Max Duration:</b> {preset['max_duration']} seconds</li>"
        
        info_text += "</ul>"
        
        if "description" in preset:
            info_text += f"<p style='margin-top: 5px;'><i>{preset['description']}</i></p>"
        
        self.platform_info_label.setText(info_text)
    
    def export_for_platform(self):
        """Export the selected clip for the chosen platform."""
        if not self.selected_clip:
            self.show_error("No clip selected")
            return
        
        # Get platform and settings
        platform = self.platform_combo.currentText().lower()
        
        # Create exporter
        self.clip_exporter = ClipExporter(
            input_path=self.selected_clip,
            output_dir=self.output_dir_input.text(),
            platform=platform
        )
        
        # Connect signals
        self.clip_exporter.progress_update.connect(self.on_export_progress)
        self.clip_exporter.status_update.connect(self.update_status)
        self.clip_exporter.export_complete.connect(self.on_export_complete)
        self.clip_exporter.error_occurred.connect(self.on_export_error)
        
        # Start export
        self.export_button.setEnabled(False)
        self.clip_exporter.start()
        
        # Save caption
        if self.caption_input.toPlainText().strip():
            caption_file = os.path.splitext(self.selected_clip)[0] + ".txt"
            try:
                with open(caption_file, 'w') as f:
                    f.write(self.caption_input.toPlainText())
            except Exception as e:
                self.log_warning(f"Could not save caption: {str(e)}")
    
    def on_stream_live(self, url):
        """Handle stream going live."""
        self.log_info(f"Stream went live: {url}")
        self.stream_url_input.setText(url)
        self.update_status(f"Stream live: {url}")
    
    def on_stream_offline(self, url):
        """Handle stream going offline."""
        self.log_info(f"Stream went offline: {url}")
        self.update_status(f"Stream offline: {url}")
    
    def on_buffer_progress(self, current, total):
        """Handle buffer progress update."""
        if total > 0:
            progress = int((current / total) * 100)
            self.buffer_progress.setValue(progress)
    
    def on_buffer_error(self, error_message):
        """Handle buffer error."""
        self.log_error(f"Buffer error: {error_message}")
        self.show_error(error_message)
        
        # Stop buffer if critical error
        if "critical" in error_message.lower() or "fatal" in error_message.lower():
            if self.stream_buffer:
                self.toggle_buffer()
    
    def on_stream_info_updated(self, info):
        """Handle stream info update from Twitch."""
        channel = info.get('channel', 'Unknown')
        qualities = info.get('qualities', [])
        
        self.log_info(f"Connected to Twitch channel: {channel}")
        if qualities:
            self.log_info(f"Available qualities: {', '.join(qualities)}")
        
        self.update_status(f"Buffering: {channel}")
    
    def on_clip_progress(self, progress):
        """Handle clip creation progress."""
        self.clip_progress.setValue(progress)
    
    def on_clip_created(self, file_path):
        """Handle successful clip creation."""
        self.log_info(f"Clip created: {file_path}")
        self.clip_progress.setValue(100)
        self.create_clip_button.setEnabled(True)
        self.update_status(f"Clip created: {os.path.basename(file_path)}")
        
        # Reload clips
        self.load_clips()
        
        # Auto-export if enabled
        if self.config_manager.get("auto_export", False):
            self.selected_clip = file_path
            self.tab_widget.setCurrentWidget(self.export_tab)
            self.prepare_export()
    
    def on_clip_error(self, error_message):
        """Handle clip creation error."""
        self.log_error(f"Clip error: {error_message}")
        self.show_error(error_message)
        self.create_clip_button.setEnabled(True)
        self.clip_progress.setValue(0)
    
    def on_clip_selected(self):
        """Handle clip selection in list."""
        selected_items = self.clips_list.selectedItems()
        
        if selected_items:
            index = self.clips_list.row(selected_items[0])
            self.selected_clip = self.clips[index]['path']
            
            # Enable action buttons
            self.play_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.delete_button.setEnabled(True)
        else:
            self.selected_clip = None
            
            # Disable action buttons
            self.play_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.delete_button.setEnabled(False)
    
    def play_clip(self):
        """Play the selected clip."""
        if self.selected_clip and os.path.exists(self.selected_clip):
            self.log_info(f"Playing clip: {self.selected_clip}")
            
            # Use default system player
            if sys.platform == "win32":
                os.startfile(self.selected_clip)
            elif sys.platform == "darwin":
                subprocess.run(["open", self.selected_clip])
            else:
                subprocess.run(["xdg-open", self.selected_clip])
    
    def analyze_clip(self):
        """Analyze the selected clip."""
        if not self.selected_clip:
            return
        
        self.log_info(f"Analyzing clip: {self.selected_clip}")
        
        # Create content analyzer
        self.content_analyzer = ContentAnalyzer(
            video_file=self.selected_clip,
            sensitivity=self.config_manager.get("viral_detection.sensitivity", 0.7)
        )
        
        # Connect signals
        self.content_analyzer.analysis_complete.connect(self.on_analysis_complete)
        self.content_analyzer.progress_update.connect(lambda p: self.update_status(f"Analyzing... {p}%"))
        self.content_analyzer.status_update.connect(self.update_status)
        
        # Start analysis
        self.content_analyzer.start()
        self.update_status("Analyzing clip...")
    
    def on_analysis_complete(self, viral_moments):
        """Handle analysis completion."""
        if viral_moments:
            self.log_info(f"Found {len(viral_moments)} viral moments")
            self.update_status(f"Analysis complete: {len(viral_moments)} viral moments found")
            
            # Show results
            moments_text = "\n".join([
                f"Moment {i+1}: {start:.1f}s - {end:.1f}s (Score: {score:.2f})"
                for i, (start, end, score) in enumerate(viral_moments)
            ])
            
            QMessageBox.information(
                self,
                "Viral Moments Found",
                f"Found {len(viral_moments)} potential viral moments:\n\n{moments_text}"
            )
        else:
            self.log_info("No viral moments found")
            self.update_status("Analysis complete: No viral moments found")
    
    def prepare_export(self):
        """Prepare clip for export."""
        if not self.selected_clip:
            self.show_error("No clip selected")
            return
        
        # Switch to export tab
        self.tab_widget.setCurrentWidget(self.export_tab)
        
        # Set selected clip
        self.selected_clip_label.setText(os.path.basename(self.selected_clip))
        
        # Load caption if available
        caption_file = os.path.splitext(self.selected_clip)[0] + ".txt"
        if os.path.exists(caption_file):
            try:
                with open(caption_file, 'r') as f:
                    self.caption_input.setText(f.read())
            except:
                self.caption_input.clear()
        else:
            self.caption_input.clear()
    
    def on_export_progress(self, progress):
        """Handle export progress."""
        self.export_progress.setValue(progress)
    
    def on_export_complete(self, success, output_path):
        """Handle export completion."""
        self.export_button.setEnabled(True)
        self.export_progress.setValue(100 if success else 0)
        
        if success:
            self.log_info(f"Export successful: {output_path}")
            self.update_status("Export successful!")
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Your clip has been exported successfully!\n\nSaved to: {output_path}"
            )
            
            # Reload clips
            self.load_clips()
        else:
            self.log_error("Export failed")
            self.update_status("Export failed")
    
    def on_export_error(self, error_message):
        """Handle export error."""
        self.log_error(f"Export error: {error_message}")
        self.show_error(error_message)
        self.export_button.setEnabled(True)
        self.export_progress.setValue(0)
    
    def delete_clip(self):
        """Delete the selected clip."""
        if not self.selected_clip:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Clip",
            f"Are you sure you want to delete:\n{os.path.basename(self.selected_clip)}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                os.remove(self.selected_clip)
                self.log_info(f"Deleted clip: {self.selected_clip}")
                self.update_status("Clip deleted")
                
                # Reload clips
                self.load_clips()
            except Exception as e:
                self.log_error(f"Failed to delete clip: {e}")
                self.show_error(f"Failed to delete clip: {e}")
    
    def analyze_video_from_url(self):
        """Analyze video from URL for viral moments without downloading full video."""
        url = self.vod_url_input.text().strip()
        if not url:
            self.show_error("Please enter a video URL")
            return
        self.vod_moments_list.clear()
        self.download_clips_button.setEnabled(False)
        self.download_all_clips_button.setEnabled(False)
        sensitivity = self.vod_sensitivity_slider.value() / 100.0
        debug_mode = self.vod_debug_checkbox.isChecked()
        self.streaming_analyzer = StreamingAnalyzer(
            video_url=url,
            sensitivity=sensitivity,
            debug_mode=debug_mode
        )
        self.streaming_analyzer.analysis_complete.connect(self.on_streaming_analysis_complete)
        self.streaming_analyzer.progress_update.connect(lambda p: self.analysis_progress.setValue(p))
        self.streaming_analyzer.status_update.connect(self.update_status)
        self.streaming_analyzer.error_occurred.connect(lambda e: self.show_error(e))
        self.streaming_analyzer.start()
        self.analyze_vod_button.setEnabled(False)

    def on_streaming_analysis_complete(self, viral_moments):
        self.analyze_vod_button.setEnabled(True)
        # Filter out invalid moments (start >= end)
        self.viral_moments = [(start, end, score) for (start, end, score) in viral_moments if end > start and (end - start) > 0.5]
        self.vod_moments_list.clear()
        if self.viral_moments:
            self.log_info(f"Found {len(self.viral_moments)} viral moments in video")
            for i, (start, end, score) in enumerate(self.viral_moments):
                duration = end - start
                item_text = f"Moment {i+1}: {start:.1f}s - {end:.1f}s ({duration:.1f}s) [Score: {int(score*100)}%]"
                self.vod_moments_list.addItem(item_text)
            self.download_clips_button.setEnabled(True)
            self.download_all_clips_button.setEnabled(True)
        else:
            self.log_info("No viral moments found in video")
            self.vod_moments_list.addItem("No viral moments detected")
            self.download_clips_button.setEnabled(False)
            self.download_all_clips_button.setEnabled(False)
        self.update_status("Analysis complete")

    def download_viral_clips(self):
        selected_items = self.vod_moments_list.selectedItems()
        if not selected_items:
            self.show_error("Please select viral moments to download")
            return
        indices = [self.vod_moments_list.row(item) for item in selected_items]
        moments = [self.viral_moments[i] for i in indices]
        self._download_clips(moments)

    def download_all_viral_clips(self):
        if not hasattr(self, 'viral_moments') or not self.viral_moments:
            self.show_error("No viral moments detected")
            return
        self._download_clips(self.viral_moments)

    def _download_clips(self, moments):
        # Filter out invalid moments again for safety
        moments = [(start, end, score) for (start, end, score) in moments if end > start and (end - start) > 0.5]
        if not moments:
            self.update_status("No valid viral moments to download.")
            self.show_error("No valid viral moments to download.")
            return
        try:
            self.update_status("Preparing to download clips...")
            self.download_progress.setValue(5)
            output_dir = self.config_manager.get("output_directory")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            total_duration = sum(end - start for start, end, _ in moments)
            downloaded_duration = 0
            video_url = self.vod_url_input.text().strip()
            # Use yt-dlp to get the direct stream URL
            try:
                import subprocess
                result = subprocess.run([
                    "yt-dlp", "--format", "best[height<=1080]", "--get-url", video_url
                ], capture_output=True, text=True, check=True)
                direct_url = result.stdout.strip().split('\n')[0]
            except Exception as e:
                self.log_error(f"yt-dlp failed: {e}")
                self.show_error("Failed to get direct video URL with yt-dlp. Make sure yt-dlp is installed and the URL is valid.")
                return
            success_count = 0
            for i, (start_time, end_time, score) in enumerate(moments):
                output_file = os.path.join(output_dir, f"viral_clip_{timestamp}_{i+1}.mp4")
                self.update_status(f"Downloading clip {i+1}/{len(moments)}...")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start_time),
                    "-i", direct_url,
                    "-t", str(end_time - start_time),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-c:a", "aac",
                    output_file
                ]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                for line in process.stderr:
                    if "time=" in line:
                        try:
                            time_str = line.split("time=")[1].split()[0]
                            parts = time_str.split(':')
                            if len(parts) == 3:
                                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                                current_seconds = h * 3600 + m * 60 + s
                                clip_duration = end_time - start_time
                                clip_progress = min(current_seconds / clip_duration, 1.0)
                                downloaded_duration = sum(end - start for start, end, _ in moments[:i])
                                progress = int(((downloaded_duration + (clip_progress * clip_duration)) / total_duration) * 90) + 5
                                self.download_progress.setValue(progress)
                        except:
                            pass
                process.wait()
                if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
                    success_count += 1
                else:
                    self.log_warning(f"Failed to download clip {i+1}")
            self.download_progress.setValue(100)
            if success_count > 0:
                self.update_status(f"Downloaded {success_count} viral clips successfully")
                self.load_clips()
                self.show_info(f"Downloaded {success_count} viral clips to {output_dir}", "Download Complete")
            else:
                self.update_status("No clips were downloaded successfully")
                self.show_error("No viral clips could be downloaded. Please check the video URL and try again.")
        except Exception as e:
            self.log_error(f"Error downloading clips: {str(e)}")
            self.show_error(f"Error: {str(e)}")

    def show_info(self, message, title="Info"):
        """Show informational message."""
        QMessageBox.information(self, title, message)

    # Utility methods
    def update_status(self, message):
        """Update status bar message."""
        self.status_label.setText(message)

    def show_error(self, message):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)

    def log_info(self, message):
        """Log info message."""
        logger.info(message)
        self.log_message.emit("INFO", message)

    def log_warning(self, message):
        """Log warning message."""
        logger.warning(message)
        self.log_message.emit("WARNING", message)

    def log_error(self, message):
        """Log error message."""
        logger.error(message)
        self.log_message.emit("ERROR", message)

    def _handle_log_message(self, level, message):
        """Handle log message display."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color code by level
        if level == "ERROR":
            color = "#ff5252"
        elif level == "WARNING":
            color = "#ffc107"
        else:
            color = "#ffffff"
        
        # Add to log text
        self.log_text.append(
            f'<span style="color: {color}">[{timestamp}] {level}: {message}</span>'
        )

    def closeEvent(self, event):
        """Enhanced close event with comprehensive cleanup."""
        # Save enhanced analytics data
        if hasattr(self, 'analytics_system'):
            self.analytics_system.save_session_data()
        
        # Stop enhanced detector
        if hasattr(self, 'enhanced_viral_detector') and self.enhanced_viral_detector and self.enhanced_viral_detector.isRunning():
            self.enhanced_viral_detector.stop()
            self.enhanced_viral_detector.wait()
        
        # Stop all running processes
        if self.stream_buffer and self.stream_buffer.isRunning():
            self.stream_buffer.stop()
            self.stream_buffer.wait()
        
        if self.stream_monitor and self.stream_monitor.isRunning():
            self.stream_monitor.stop()
            self.stream_monitor.wait()
        
        # Save settings
        self.save_settings()
        
        # Stop temp file manager
        self.temp_manager.stop_cleanup_timer()
        
        event.accept()

    def save_settings(self):
        """Save settings to configuration."""
        self.config_manager.set("buffer_duration", self.default_buffer_spin.value())
        self.config_manager.set("auto_export", self.auto_export_check.isChecked())
        self.config_manager.set("auto_monitor", self.auto_monitor_check.isChecked())
        self.config_manager.set("format", self.format_combo.currentText())
        self.config_manager.set("resolution", self.resolution_combo.currentText())
        self.config_manager.set("output_directory", self.output_dir_input.text())
        self.log_info("Settings saved")
        self.update_status("Settings saved")

    def on_ad_status_update(self, status):
        """Handle ad status updates from the buffer."""
        self.ad_status_label.setText(status)
        
        # Color code the status
        if "Ads playing" in status or "waiting" in status.lower():
            self.ad_status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.log_info("Ads detected - waiting for content to resume")
        elif "Recording active" in status:
            self.ad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "resumed" in status.lower():
            self.ad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.log_info("Content resumed - recording active")
        else:
            self.ad_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")

    def _on_sensitivity_slider_changed(self, value):
        """Handle sensitivity slider change with debouncing."""
        self.sensitivity_value.setText(f"{value}%")
        self.sensitivity_timer.stop()
        self.sensitivity_timer.start(500)  # Wait 500ms after user stops
    
    def _apply_sensitivity_change(self):
        """Apply the sensitivity change after debounce period."""
        value = self.sensitivity_slider.value()
        self.log_info(f"Applying enhanced sensitivity change: {value}%")
        self.update_enhanced_sensitivity(value)


# Enhanced viral moment manager
class EnhancedViralMomentManager(ViralMomentManager):
    """Enhanced viral moment manager with analytics integration."""
    
    def __init__(self, max_moments=20):
        super().__init__(max_moments)
        self.moment_ratings = {}  # moment_id -> user_rating
    
    def rate_moment(self, moment_index, rating):
        """Rate a moment for learning purposes."""
        if 0 <= moment_index < len(self.moments):
            self.moment_ratings[moment_index] = rating
            return True
        return False
    
    def get_moment_rating(self, moment_index):
        """Get user rating for a moment."""
        return self.moment_ratings.get(moment_index, None)
    
    def get_analytics_summary(self):
        """Get analytics summary of detected moments."""
        if not self.moments:
            return {}
        
        return {
            'total_moments': len(self.moments),
            'avg_confidence': np.mean([m['score'] for m in self.moments]),
            'context_distribution': Counter([m.get('context', 'unknown') for m in self.moments]),
            'rated_moments': len(self.moment_ratings),
            'avg_rating': np.mean(list(self.moment_ratings.values())) if self.moment_ratings else 0
        }


# Enhanced clip creator with metadata
class EnhancedClipCreator(ClipCreator):
    """Enhanced clip creator that embeds viral moment metadata."""
    
    def __init__(self, stream_buffer, start_time_ago, duration, output_path, format_type, moment_metadata):
        super().__init__(stream_buffer, start_time_ago, duration, output_path, format_type)
        self.moment_metadata = moment_metadata
    
    def run(self):
        """Create clip with enhanced metadata embedding."""
        # Run normal clip creation
        super().run()
        
        # After clip creation, embed metadata
        if os.path.exists(self.output_path):
            self._embed_metadata()
    
    def _embed_metadata(self):
        """Embed viral moment metadata into clip file."""
        try:
            metadata_file = self.output_path.replace('.mp4', '_metadata.json')
            
            metadata = {
                'viral_detection': {
                    'confidence': self.moment_metadata.get('confidence', self.moment_metadata.get('score', 0)),
                    'context': self.moment_metadata.get('context', 'unknown'),
                    'social_proof': self.moment_metadata.get('social_proof', {}),
                    'signals': self.moment_metadata.get('signals', {}),
                    'uniqueness': self.moment_metadata.get('uniqueness', 0),
                    'detected_at': self.moment_metadata.get('detected_at', time.time()),
                    'description': self.moment_metadata.get('description', '')
                },
                'clip_info': {
                    'start_time': self.moment_metadata.get('clip_start', self.start_time_ago),
                    'duration': self.duration,
                    'created_at': time.time(),
                    'format': self.format_type
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Embedded viral metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error embedding metadata: {e}")


def main():
    """Application entry point."""
    # Create log directory in unified temp folder
    log_dir = os.path.join(BASE_TEMP_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # Set up file logging
    log_file = os.path.join(log_dir, f"beastclipper_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    
    # Create and show main window
    window = BeastClipperApp()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
