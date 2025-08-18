# backend/app.py
"""
Main Flask application for Contextual Music Triggering System
Handles audio streaming, emotion detection, and music playback
"""

import os
import sys
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import numpy as np
from loguru import logger

# Import custom modules
from core.audio_processor import AudioProcessor
from core.youtube_player import YouTubePlayer
from core.xai_explainer import XAIExplainer
from models.emotion_detector import EmotionDetector
from models.music_retriever import MusicRetriever
from models.conversation_analyzer import ConversationAnalyzer
from utils.config import Config
from utils.logger import setup_logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Enable CORS
CORS(app)

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Setup logging
setup_logging(debug=os.getenv('DEBUG_MODE', 'False').lower() == 'true')

# Global components
class SystemComponents:
    """Container for all system components"""
    def __init__(self):
        logger.info("Initializing system components...")
        
        self.config = Config()
        self.audio_processor = None
        self.emotion_detector = None
        self.music_retriever = None
        self.conversation_analyzer = None
        self.youtube_player = None
        self.xai_explainer = None
        
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.last_trigger_time = 0
        self.conversation_history = []
        self.current_emotion_state = "neutral"
        self.emotion_stability = 0.5
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components with error handling"""
        try:
            # Initialize audio processor
            logger.info("Loading audio processor...")
            self.audio_processor = AudioProcessor(
                sample_rate=int(os.getenv('AUDIO_SAMPLE_RATE', 16000))
            )
            
            # Initialize emotion detector
            logger.info("Loading emotion detector (Mistral-7B)...")
            self.emotion_detector = EmotionDetector(
                model_name=os.getenv('MODEL_NAME', 'mistralai/Mistral-7B-Instruct-v0.3')
            )
            
            # Initialize music retriever
            logger.info("Loading music retriever...")
            self.music_retriever = MusicRetriever(
                dataset_path='data/synthetic_music_dataset.json',
                embedding_model=os.getenv('EMBEDDING_MODEL', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct')
            )
            
            # Initialize conversation analyzer
            logger.info("Loading conversation analyzer...")
            self.conversation_analyzer = ConversationAnalyzer()
            
            # Initialize YouTube player
            logger.info("Setting up YouTube player...")
            self.youtube_player = YouTubePlayer(
                api_key=os.getenv('YOUTUBE_API_KEY')
            )
            
            # Initialize XAI explainer
            if os.getenv('ENABLE_XAI', 'True').lower() == 'true':
                logger.info("Loading XAI explainer...")
                self.xai_explainer = XAIExplainer()
            
            logger.success("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

# Initialize system
system = SystemComponents()

# Routes
@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'audio': system.audio_processor is not None,
            'emotion': system.emotion_detector is not None,
            'music': system.music_retriever is not None,
            'youtube': system.youtube_player is not None
        }
    })

@app.route('/api/system/status')
def system_status():
    """Get current system status"""
    return jsonify({
        'is_listening': system.is_listening,
        'current_emotion': system.current_emotion_state,
        'emotion_stability': system.emotion_stability,
        'conversation_history_size': len(system.conversation_history),
        'last_trigger': system.last_trigger_time
    })

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to music trigger system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    if system.is_listening:
        system.is_listening = False

@socketio.on('start_listening')
def handle_start_listening():
    """Start audio capture and processing"""
    if not system.is_listening:
        system.is_listening = True
        logger.info("Starting audio capture...")
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=process_audio_stream)
        audio_thread.daemon = True
        audio_thread.start()
        
        emit('listening_started', {'status': 'active'})
        logger.success("Audio capture started")
    else:
        emit('error', {'message': 'Already listening'})

@socketio.on('stop_listening')
def handle_stop_listening():
    """Stop audio capture"""
    if system.is_listening:
        system.is_listening = False
        logger.info("Stopping audio capture...")
        emit('listening_stopped', {'status': 'inactive'})
    else:
        emit('error', {'message': 'Not currently listening'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk from client"""
    if system.is_listening:
        try:
            # Convert base64 audio to numpy array
            audio_data = np.frombuffer(data['audio'], dtype=np.float32)
            system.audio_queue.put(audio_data)
            
            logger.debug(f"Received audio chunk: {len(audio_data)} samples")
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            emit('error', {'message': f'Audio processing error: {str(e)}'})

def process_audio_stream():
    """Main audio processing loop"""
    logger.info("Audio processing thread started")
    
    audio_buffer = []
    buffer_duration = int(os.getenv('RECORDING_DURATION', 5))
    sample_rate = int(os.getenv('AUDIO_SAMPLE_RATE', 16000))
    buffer_size = buffer_duration * sample_rate
    
    while system.is_listening:
        try:
            # Collect audio chunks
            if not system.audio_queue.empty():
                chunk = system.audio_queue.get(timeout=0.1)
                audio_buffer.extend(chunk)
                
                # Process when buffer is full
                if len(audio_buffer) >= buffer_size:
                    logger.debug("Processing audio buffer...")
                    
                    # Convert to numpy array
                    audio_array = np.array(audio_buffer[:buffer_size], dtype=np.float32)
                    
                    # Process audio
                    process_conversation(audio_array)
                    
                    # Keep last 20% of buffer for continuity
                    audio_buffer = audio_buffer[int(buffer_size * 0.8):]
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
            socketio.emit('error', {'message': f'Processing error: {str(e)}'})
    
    logger.info("Audio processing thread stopped")

def process_conversation(audio_data):
    """Process audio to detect emotion and trigger music"""
    try:
        # Transcribe audio
        logger.debug("Transcribing audio...")
        transcript = system.audio_processor.transcribe(audio_data)
        
        if not transcript or len(transcript.strip()) < 3:
            return
        
        logger.info(f"Transcript: {transcript}")
        
        # Send transcript to frontend
        socketio.emit('transcript', {
            'text': transcript,
            'timestamp': datetime.now().isoformat()
        })
        
        # Analyze conversation context
        context = system.conversation_analyzer.analyze(
            transcript, 
            system.conversation_history
        )
        
        # Detect emotion
        emotion_result = system.emotion_detector.detect(
            transcript,
            context
        )
        
        logger.info(f"Detected emotion: {emotion_result['emotion']} "
                   f"(confidence: {emotion_result['confidence']:.2%})")
        
        # Update emotion state
        update_emotion_state(emotion_result)
        
        # Send emotion update to frontend
        socketio.emit('emotion_update', {
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'intensity': emotion_result.get('intensity', 0.5),
            'stability': system.emotion_stability
        })
        
        # Check if we should trigger music
        if should_trigger_music(emotion_result):
            trigger_music(context, emotion_result)
        
        # Update conversation history
        system.conversation_history.append({
            'text': transcript,
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        if len(system.conversation_history) > int(os.getenv('MAX_CONVERSATION_HISTORY', 10)):
            system.conversation_history.pop(0)
            
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        socketio.emit('error', {'message': f'Processing error: {str(e)}'})

def update_emotion_state(emotion_result):
    """Update global emotion state with stability tracking"""
    new_emotion = emotion_result['emotion']
    confidence = emotion_result['confidence']
    
    if system.current_emotion_state == new_emotion:
        # Increase stability for consistent emotion
        system.emotion_stability = min(1.0, system.emotion_stability + 0.1)
    else:
        # Check if we should switch emotions
        if confidence > float(os.getenv('EMOTION_THRESHOLD', 0.6)) and \
           system.emotion_stability < 0.5:
            system.current_emotion_state = new_emotion
            system.emotion_stability = confidence * 0.5
        else:
            # Decrease stability
            system.emotion_stability = max(0.0, system.emotion_stability - 0.1)
    
    logger.debug(f"Emotion state: {system.current_emotion_state}, "
                f"Stability: {system.emotion_stability:.2f}")

def should_trigger_music(emotion_result):
    """Determine if music should be triggered"""
    current_time = time.time()
    min_interval = int(os.getenv('MIN_TRIGGER_INTERVAL', 30))
    
    # Check time since last trigger
    if current_time - system.last_trigger_time < min_interval:
        return False
    
    # Check emotion confidence and intensity
    if emotion_result['confidence'] < float(os.getenv('MIN_CONFIDENCE', 0.5)):
        return False
    
    if emotion_result.get('intensity', 0) < 0.4:
        return False
    
    # Don't trigger for neutral emotion
    if emotion_result['emotion'] == 'neutral':
        return False
    
    return True

def trigger_music(context, emotion_result):
    """Trigger music playback based on emotion"""
    try:
        logger.info("Triggering music selection...")
        
        # Retrieve matching songs
        songs = system.music_retriever.retrieve(
            context=context,
            emotion=emotion_result['emotion'],
            top_k=5
        )
        
        if not songs:
            logger.warning("No matching songs found")
            return
        
        # Select best song
        selected_song = songs[0]
        logger.info(f"Selected: {selected_song['title']} - {selected_song['artist']}")
        
        # Generate explanation if XAI is enabled
        explanation = None
        if system.xai_explainer:
            explanation = system.xai_explainer.explain(
                context=context,
                emotion=emotion_result,
                song=selected_song,
                alternatives=songs[1:4]
            )
        
        # Send music selection to frontend
        socketio.emit('music_selected', {
            'song': selected_song,
            'emotion': emotion_result['emotion'],
            'confidence': emotion_result['confidence'],
            'explanation': explanation,
            'alternatives': songs[1:4]
        })
        
        # Search and play on YouTube
        youtube_url = system.youtube_player.search_and_get_url(
            f"{selected_song['title']} {selected_song['artist']}"
        )
        
        if youtube_url:
            # Send YouTube URL to frontend for playback
            socketio.emit('play_music', {
                'url': youtube_url,
                'duration': int(os.getenv('MUSIC_DURATION', 45)),
                'song': selected_song
            })
            
            # Update last trigger time
            system.last_trigger_time = time.time()
            
            logger.success(f"Music triggered successfully: {youtube_url}")
        else:
            logger.warning("Could not find song on YouTube")
            socketio.emit('error', {'message': 'Song not found on YouTube'})
            
    except Exception as e:
        logger.error(f"Error triggering music: {e}")
        socketio.emit('error', {'message': f'Music trigger error: {str(e)}'})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('temp').mkdir(exist_ok=True)
    Path('model_cache').mkdir(exist_ok=True)
    
    # Run the application
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    
    socketio.run(app, host=host, port=port, debug=debug)