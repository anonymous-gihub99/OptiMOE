# antashiri_main.py
"""
Antashiri - Main Setup and Run Script
Complete system initialization and management
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import threading

from loguru import logger


class AntashiriSystem:
    """
    Main Antashiri system manager
    Handles setup, initialization, and running of all components
    """
    
    def __init__(self):
        """Initialize Antashiri system manager"""
        self.project_root = Path(__file__).parent
        self.is_setup_complete = False
        self.server_process = None
        
        # ASCII Art Logo
        self.logo = """
        ╔═══════════════════════════════════════════════════════════╗
        ║                                                           ║
        ║     █████╗ ███╗   ██╗████████╗ █████╗ ███████╗██╗  ██╗  ║
        ║    ██╔══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔════╝██║  ██║  ║
        ║    ███████║██╔██╗ ██║   ██║   ███████║███████╗███████║  ║
        ║    ██╔══██║██║╚██╗██║   ██║   ██╔══██║╚════██║██╔══██║  ║
        ║    ██║  ██║██║ ╚████║   ██║   ██║  ██║███████║██║  ██║  ║
        ║    ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝  ║
        ║                                                           ║
        ║    IRIS - Intelligent Rhythmic Interaction System        ║
        ║    LLM-Enhanced Contextual Music Triggering with XAI     ║
        ╚═══════════════════════════════════════════════════════════╝
        """
    
    def print_logo(self):
        """Print system logo"""
        print("\033[95m" + self.logo + "\033[0m")
    
    def check_requirements(self) -> bool:
        """Check if all requirements are installed"""
        
        print("\n📋 Checking requirements...")
        
        required_packages = [
            'torch', 'transformers', 'whisper', 'flask',
            'sentence_transformers', 'faiss', 'pytube', 'loguru'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        print("\n✅ All requirements satisfied!")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        
        print("\n📁 Setting up directories...")
        
        directories = [
            'data', 'models', 'core', 'backend', 'frontend/static',
            'frontend/templates', 'utils', 'scripts', 'logs',
            'temp', 'cache', 'config'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path}/")
        
        print("\n✅ Directory structure created!")
    
    def generate_dataset(self) -> bool:
        """Generate synthetic dataset"""
        
        print("\n🎵 Generating synthetic dataset...")
        
        dataset_path = self.project_root / 'data' / 'synthetic_music_dataset.json'
        
        if dataset_path.exists():
            print("  ℹ️  Dataset already exists")
            response = input("  Regenerate? (y/n): ").lower()
            if response != 'y':
                return True
        
        try:
            # Import and run generator
            from scripts.generate_synthetic_dataset import SyntheticMusicDatasetGenerator
            
            generator = SyntheticMusicDatasetGenerator(seed=42)
            dataset = generator.generate_dataset(num_songs=20000)
            generator.save_dataset(dataset, str(dataset_path))
            
            print(f"\n✅ Dataset generated: {len(dataset['songs'])} songs")
            return True
            
        except Exception as e:
            print(f"\n❌ Dataset generation failed: {e}")
            return False
    
    def initialize_models(self) -> bool:
        """Initialize and test models"""
        
        print("\n🤖 Initializing models...")
        
        try:
            # Test imports
            print("  Loading Emotion Detector...")
            from models.emotion_detector import EmotionDetector
            
            print("  Loading Music Retriever...")
            from models.music_retriever import MusicRetriever
            
            print("  Loading Audio Processor...")
            from core.audio_processor import AudioProcessor
            
            print("  Loading Conversation Analyzer...")
            from models.conversation_analyzer import ConversationAnalyzer
            
            print("  Loading XAI Explainer...")
            from core.xai_explainer import XAIExplainer
            
            print("\n✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Model initialization failed: {e}")
            return False
    
    def create_config(self):
        """Create configuration file"""
        
        print("\n⚙️  Creating configuration...")
        
        config_path = self.project_root / 'config' / 'antashiri.json'
        
        config = {
            'system': {
                'name': 'Antashiri',
                'version': '1.0.0',
                'debug': True
            },
            'models': {
                'emotion': 'mistralai/Mistral-7B-Instruct-v0.3',
                'whisper': 'base',
                'embeddings': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 5000
            },
            'audio': {
                'sample_rate': 16000,
                'duration': 5
            },
            'music': {
                'duration': 45,
                'min_interval': 30
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✅ Configuration saved to {config_path}")
    
    def create_env_file(self):
        """Create .env file template"""
        
        print("\n🔐 Creating environment file...")
        
        env_path = self.project_root / '.env'
        
        if env_path.exists():
            print("  ℹ️  .env file already exists")
            return
        
        env_content = """# Antashiri Environment Configuration

# Model Settings
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
WHISPER_MODEL=base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Keys (Optional)
YOUTUBE_API_KEY=your_youtube_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Server Settings
HOST=0.0.0.0
PORT=5000
DEBUG_MODE=True

# Audio Settings
AUDIO_SAMPLE_RATE=16000
RECORDING_DURATION=5

# Music Settings
MUSIC_DURATION=45
MIN_TRIGGER_INTERVAL=30

# Emotion Detection
EMOTION_THRESHOLD=0.6
MIN_CONFIDENCE=0.5

# Paths
DATA_PATH=./data
LOG_PATH=./logs
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"  ✅ Environment file created: {env_path}")
        print("  ⚠️  Remember to add your API keys!")
    
    def test_system(self) -> bool:
        """Run system tests"""
        
        print("\n🧪 Running system tests...")
        
        try:
            # Test emotion detection
            print("\n  Testing Emotion Detection...")
            from models.emotion_detector import EmotionDetector
            detector = EmotionDetector()
            result = detector.detect("I'm so happy today!")
            print(f"    Result: {result['emotion']} ({result['confidence']:.2%})")
            
            # Test music retrieval
            print("\n  Testing Music Retrieval...")
            from models.music_retriever import MusicRetriever
            retriever = MusicRetriever()
            songs = retriever.retrieve(
                {'keywords': ['happy']}, 
                'happy', 
                top_k=3
            )
            if songs:
                print(f"    Found {len(songs)} songs")
                print(f"    Top: {songs[0]['title']} - {songs[0]['artist']}")
            
            print("\n✅ System tests passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ System tests failed: {e}")
            return False
    
    def start_server(self):
        """Start the Flask server"""
        
        print("\n🚀 Starting Antashiri server...")
        
        try:
            # Start server in subprocess
            cmd = [sys.executable, str(self.project_root / 'backend' / 'app.py')]
            self.server_process = subprocess.Popen(cmd)
            
            print("\n✅ Server started!")
            print("🌐 Open http://localhost:5000 in your browser")
            print("\n📝 Instructions:")
            print("  1. Click the microphone button to start")
            print("  2. Allow microphone access when prompted")
            print("  3. Start speaking naturally")
            print("  4. Music will play based on your emotional state")
            print("\n⚠️  Press Ctrl+C to stop the server")
            
            # Wait for server to stop
            self.server_process.wait()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down server...")
            if self.server_process:
                self.server_process.terminate()
            print("✅ Server stopped")
        except Exception as e:
            print(f"\n❌ Server failed to start: {e}")
    
    def run_notebook(self):
        """Launch Jupyter notebook"""
        
        print("\n📓 Launching Jupyter Notebook...")
        
        try:
            subprocess.run(['jupyter', 'notebook', 'Antashiri_Complete_System.ipynb'])
        except Exception as e:
            print(f"❌ Failed to launch notebook: {e}")
            print("Run manually: jupyter notebook Antashiri_Complete_System.ipynb")
    
    def setup(self):
        """Run complete setup process"""
        
        self.print_logo()
        print("\n🔧 ANTASHIRI SETUP")
        print("="*60)
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Setup failed: Missing requirements")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Create config files
        self.create_config()
        self.create_env_file()
        
        # Generate dataset
        if not self.generate_dataset():
            print("\n❌ Setup failed: Dataset generation error")
            return False
        
        # Initialize models
        if not self.initialize_models():
            print("\n❌ Setup failed: Model initialization error")
            return False
        
        # Run tests
        if not self.test_system():
            print("\n⚠️  Some tests failed, but setup complete")
        
        self.is_setup_complete = True
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        
        return True
    
    def run(self, mode: str = 'server'):
        """Run Antashiri in specified mode"""
        
        if not self.is_setup_complete:
            print("⚠️  Running setup first...")
            if not self.setup():
                return
        
        if mode == 'server':
            self.start_server()
        elif mode == 'notebook':
            self.run_notebook()
        elif mode == 'test':
            self.test_system()
        else:
            print(f"❌ Unknown mode: {mode}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Antashiri - Contextual Music System')
    parser.add_argument('command', choices=['setup', 'run', 'test', 'notebook'],
                       help='Command to execute')
    parser.add_argument('--mode', default='server',
                       choices=['server', 'notebook', 'test'],
                       help='Run mode (for run command)')
    
    args = parser.parse_args()
    
    # Initialize system
    antashiri = AntashiriSystem()
    
    # Execute command
    if args.command == 'setup':
        antashiri.setup()
    elif args.command == 'run':
        antashiri.run(mode=args.mode)
    elif args.command == 'test':
        antashiri.test_system()
    elif args.command == 'notebook':
        antashiri.run_notebook()


if __name__ == "__main__":
    # If no arguments, run setup and server
    if len(sys.argv) == 1:
        antashiri = AntashiriSystem()
        if antashiri.setup():
            print("\n" + "="*60)
            response = input("Start server now? (y/n): ").lower()
            if response == 'y':
                antashiri.run('server')
    else:
        main()