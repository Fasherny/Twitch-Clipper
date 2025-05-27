#!/usr/bin/env python3
"""
Quick Test Script for BeastClipper Components
Tests each component individually to identify issues
"""

import sys
import time
import traceback

def test_imports():
    """Test if all imports work."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 - OK")
    except ImportError as e:
        print(f"❌ PyQt5 - FAILED: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV - OK")
    except ImportError as e:
        print(f"❌ OpenCV - FAILED: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy - OK")
    except ImportError as e:
        print(f"❌ NumPy - FAILED: {e}")
        return False
    
    # NO MORE SELENIUM NEEDED! 🎉
    print("✅ Selenium - SKIPPED (Using IRC instead - no browser needed!)")
    
    try:
        # Test your existing modules
        from social_proof_analyzer import SocialProofAnalyzer
        print("✅ Social Proof Analyzer - OK")
    except ImportError as e:
        print(f"❌ Social Proof Analyzer - FAILED: {e}")
        return False
    
    try:
        # Test the new IRC detector
        from twitch_irc_simple import ProfessionalTwitchIRCDetector
        print("✅ Professional IRC Detector - OK (no browser needed!)")
    except ImportError as e:
        print(f"❌ Professional IRC Detector - FAILED: {e}")
        print("   Make sure you created 'twitch_irc_simple.py' with the IRC detector code")
        return False
    
    try:
        from ultimate_viral_system import UltimateViralDetector
        print("✅ Ultimate Viral System - OK")
    except ImportError as e:
        print(f"❌ Ultimate Viral System - FAILED: {e}")
        return False
    
    try:
        from stream import StreamBuffer
        print("✅ Stream Buffer - OK")
    except ImportError as e:
        print(f"❌ Stream Buffer - FAILED: {e}")
        return False
    
    print("🎉 All imports successful! (IRC-powered, no browser dependency!)")
    return True

def test_social_proof_analyzer():
    """Test social proof analyzer with dummy data."""
    print("\n🧪 Testing Social Proof Analyzer...")
    
    try:
        from social_proof_analyzer import SocialProofAnalyzer
        
        analyzer = SocialProofAnalyzer()
        
        # Test with dummy chat messages
        dummy_messages = [
            {'username': 'user1', 'text': 'poggers that was insane!', 'timestamp': time.time()},
            {'username': 'user2', 'text': 'clip that omegalul', 'timestamp': time.time()},
            {'username': 'mod_user', 'text': 'holy shit no way', 'timestamp': time.time()},
            {'username': 'user3', 'text': 'YOOOOO FIRE', 'timestamp': time.time()}
        ]
        
        result = analyzer.analyze_social_signals(dummy_messages, time.time())
        
        print(f"✅ Analysis complete:")
        print(f"   - Unique chatters: {result['unique_chatters']}")
        print(f"   - Total score: {result['total_score']:.3f}")
        print(f"   - VIP engagement: {result['vip_engagement']['score']:.3f}")
        print(f"   - Message quality: {result['message_quality']['total_score']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Social Proof Analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_stream_buffer_basic():
    """Test if stream buffer can be created (without actually connecting)."""
    print("\n🧪 Testing Stream Buffer Creation...")
    
    try:
        from stream import StreamBuffer
        from config import TempFileManager, ConfigManager
        
        # Create minimal config
        config = ConfigManager()
        temp_manager = TempFileManager(config)
        
        # Try to create buffer (don't start it)
        buffer = StreamBuffer(
            stream_url="https://twitch.tv/test",
            buffer_duration=60,
            temp_manager=temp_manager
        )
        
        print("✅ Stream Buffer created successfully")
        print(f"   - Temp dir: {buffer.temp_dir}")
        print(f"   - Buffer duration: {buffer.buffer_duration}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Stream Buffer creation failed: {e}")
        traceback.print_exc()
        return False

def test_ffmpeg():
    """Test if FFmpeg is available."""
    print("\n🧪 Testing FFmpeg...")
    
    try:
        import subprocess
        
        # Test FFmpeg
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ FFmpeg - OK")
            version_line = result.stdout.split('\n')[0]
            print(f"   - {version_line}")
        else:
            print("❌ FFmpeg - Not working properly")
            return False
        
        # Test FFprobe
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ FFprobe - OK")
        else:
            print("❌ FFprobe - Not found")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ FFmpeg/FFprobe - Not found in PATH")
        print("   Download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"❌ FFmpeg test failed: {e}")
        return False

def test_streamlink():
    """Test if streamlink is available."""
    print("\n🧪 Testing Streamlink...")
    
    try:
        import subprocess
        
        result = subprocess.run(['streamlink', '--version'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Streamlink - OK")
            version_line = result.stdout.strip()
            print(f"   - {version_line}")
            return True
        else:
            print("❌ Streamlink - Not working")
            return False
        
    except FileNotFoundError:
        print("❌ Streamlink - Not found")
        print("   Install with: pip install streamlink")
        return False
    except Exception as e:
        print(f"❌ Streamlink test failed: {e}")
        return False

def test_irc_connection():
    """Test IRC connection to Twitch (replaces Chrome WebDriver test)."""
    print("\n🧪 Testing IRC Connection to Twitch...")
    
    try:
        import socket
        
        # Test basic IRC connection to Twitch
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        try:
            sock.connect(('irc.chat.twitch.tv', 6667))
            print("✅ IRC Connection - OK")
            print("   - Successfully connected to irc.chat.twitch.tv:6667")
            
            # Test basic IRC protocol
            sock.send(b"NICK justinfan12345\r\n")
            sock.send(b"USER justinfan12345 0 * :justinfan12345\r\n")
            
            # Wait for response
            response = sock.recv(1024).decode('utf-8', errors='ignore')
            if '001' in response or 'Welcome' in response:
                print("   - IRC protocol handshake successful")
            
            sock.close()
            return True
            
        except socket.timeout:
            print("❌ IRC Connection - TIMEOUT")
            print("   - Could not connect to Twitch IRC within 5 seconds")
            return False
        except ConnectionRefusedError:
            print("❌ IRC Connection - REFUSED")
            print("   - Twitch IRC server refused connection")
            return False
            
    except Exception as e:
        print(f"❌ IRC Connection - FAILED: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def test_professional_irc_detector():
    """Test the professional IRC detector."""
    print("\n🧪 Testing Professional IRC Detector...")
    
    try:
        from twitch_irc_simple import ProfessionalTwitchIRCDetector
        
        # Create detector (don't start it)
        detector = ProfessionalTwitchIRCDetector(
            channel_name="test_channel",
            sensitivity=0.7,
            use_oauth=False
        )
        
        print("✅ IRC Detector Creation - OK")
        print(f"   - Channel: #{detector.channel_name}")
        print(f"   - Sensitivity: {detector.sensitivity}")
        print(f"   - OAuth: {detector.use_oauth}")
        print(f"   - Nickname: {detector.nickname}")
        
        # Test threshold calculation
        print(f"   - Message spike threshold: {detector.thresholds['message_spike_multiplier']:.2f}")
        print(f"   - Viral keyword threshold: {detector.thresholds['viral_keyword_score']:.2f}")
        
        # Test keyword detection
        test_message = {
            'username': 'test_user',
            'text': 'omegalul that was insane! clip it!',
            'timestamp': time.time(),
            'tags': {},
            'user_type': 'regular',
            'caps_ratio': 0.1,
            'message_length': 30
        }
        
        viral_score = detector._calculate_message_viral_score(test_message)
        print(f"   - Test message viral score: {viral_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Professional IRC Detector test failed: {e}")
        traceback.print_exc()
        return False

def test_viral_detection_basic():
    """Test basic viral detection without streams."""
    print("\n🧪 Testing Viral Detection Logic...")
    
    try:
        # Test the enhanced integration if it exists
        try:
            from enhanced_viral_integration import EnhancedSignalFusion
            
            fusion = EnhancedSignalFusion(0.7)
            
            # Test with dummy signal data
            dummy_signals = {
                'timestamp': time.time(),
                'chat_signals': [{'score': 0.8, 'timestamp': time.time()}],
                'video_signals': [{'score': 0.6, 'timestamp': time.time()}],
                'social_signals': [{'score': 1.2, 'timestamp': time.time()}],
                'momentum_signals': [{'score': 0.7, 'timestamp': time.time()}]
            }
            
            result = fusion.process_signals(dummy_signals, 'gaming', 0.7)
            
            if result:
                print(f"✅ Enhanced Signal Fusion - OK")
                print(f"   - Confidence: {result['confidence']:.3f}")
                print(f"   - Sync quality: {result['sync_quality']:.3f}")
            else:
                print("✅ Enhanced Signal Fusion - OK (no detection, normal)")
            
        except ImportError:
            print("⚠️  Enhanced integration not found (you need to create enhanced_viral_integration.py)")
        
        # Test existing ultimate viral system
        from ultimate_viral_system import ViralMoment
        
        # Create test viral moment
        test_moment = ViralMoment(
            timestamp=time.time(),
            confidence=0.85,
            clip_start=0,
            clip_end=20,
            signals={'chat': 0.8, 'video': 0.6},
            social_proof={'unique_chatters': 5},
            context='gaming',
            description='Test moment',
            keywords=['test'],
            user_reactions=['hype'],
            momentum_score=0.7,
            uniqueness_score=0.8,
            detected_at=time.time(),
            streamer_profile_match=0.9
        )
        
        print("✅ ViralMoment creation - OK")
        print(f"   - Confidence: {test_moment.confidence}")
        print(f"   - Context: {test_moment.context}")
        
        return True
        
    except Exception as e:
        print(f"❌ Viral Detection test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests (IRC-powered, no browser needed!)."""
    print("🚀 BeastClipper Component Test Suite (IRC Edition)")
    print("=" * 55)
    
    tests = [
        ("Imports", test_imports),
        ("FFmpeg", test_ffmpeg),
        ("Streamlink", test_streamlink),
        ("IRC Connection", test_irc_connection),  # Replaces Chrome WebDriver
        ("Professional IRC Detector", test_professional_irc_detector),  # New test
        ("Social Proof Analyzer", test_social_proof_analyzer),
        ("Stream Buffer", test_stream_buffer_basic),
        ("Viral Detection", test_viral_detection_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} - CRASHED: {e}")
    
    print("\n" + "=" * 55)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your IRC-powered app should work perfectly!")
    elif passed >= total - 2:
        print("⚠️  Minor issues found, but IRC-powered app should mostly work")
    else:
        print("❌ Major issues found. Need to fix dependencies.")
    
    print(f"\n💡 Next steps:")
    if passed < total:
        print("1. Fix the failed tests above")
        print("2. Install missing dependencies")
    print("3. Create enhanced_viral_integration.py")
    print("4. Create twitch_irc_simple.py (Professional IRC Detector)")
    print("5. Update main.py with integration code")
    print("6. Test with a real Twitch stream")
    print("\n🚀 ADVANTAGE: No browser dependencies! IRC is faster and more reliable!")
    print("✅ No Selenium, Chrome, or WebDriver needed!")
    print("✅ Direct IRC connection to Twitch chat!")
    print("✅ Professional-grade chat monitoring!")

if __name__ == "__main__":
    main()