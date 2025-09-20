
"""
Real-Time Visualizer (v5.12.11 - True Real-Time Inference)

True real-time EMG-to-angle prediction with continuous inference.
Features:
- 512-sample rolling buffer sliding at 1kHz
- Continuous model inference as fast as possible
- Producer-consumer pattern for optimal performance
- Ground truth updates at 1kHz, predictions as fast as inference completes

Joint Mapping (matching hand_tracking.py):
- angle1: Interior angle at DIP joint (TIP-DIP-PIP)
- angle2: Interior angle at PIP joint (DIP-PIP-MCP)
- Display: Bend angles (180° - interior angle)
"""

import math
import time
import cv2
import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import joblib
import threading
import queue
import serial
import struct
from collections import deque
from typing import Tuple, Optional


class HybridCNNLSTM(nn.Module):
    """
    EMG-Delta optimized CNN-LSTM model for angle prediction - v5.12.11
    Designed for 1kHz sampling, 512ms context, 6 EMG deltas + 2 angles
    """
    def __init__(self, config):
        super(HybridCNNLSTM, self).__init__()
        
        self.config = config
        self.input_features = config['input_features']  # 8 total
        self.emg_delta_channels = config['emg_delta_channels']  # 6
        self.angle_channels = config['angle_channels']  # 2
        self.output_features = config['output_features']  # 2
        self.sequence_length = config['sequence_length']  # 512
        
        # CNN Layer (matching actual saved model architecture)
        self.cnn = nn.Conv1d(
            in_channels=self.input_features,  # 8
            out_channels=config['cnn_filters'],  # 64
            kernel_size=config['cnn_kernel_size'],  # 5
            padding=config['cnn_kernel_size']//2,  # 2
            bias=False
        )
        self.cnn_bn = nn.BatchNorm1d(config['cnn_filters'])
        self.relu = nn.ReLU(inplace=True)
        
        # Optimized Pooling: Single operation 512 → 32 directly
        self.pooling = nn.MaxPool1d(
            kernel_size=config['pool_size'] * config['pool_size'],  # 4*4 = 16
            stride=config['pool_size'] * config['pool_size']
        )
        
        # LSTM Processing (matching actual config)
        self.cnn_output_size = config['cnn_filters']  # 64
        self.final_timesteps = config['final_cnn_timesteps']  # 32
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,  # 64
            hidden_size=config['lstm_hidden_size'],  # 128
            num_layers=config['lstm_num_layers'],  # 1
            batch_first=True,
            bidirectional=False
        )
        
        # Vectorized Temporal Attention (matching actual architecture)
        self.attention_projection = nn.Linear(config['lstm_hidden_size'], config['lstm_hidden_size'] // 2, bias=False)  # 128->64
        self.attention_output = nn.Linear(config['lstm_hidden_size'] // 2, 1, bias=False)  # 64->1
        self.attention_tanh = nn.Tanh()
        
        # Dense Layers (matching actual architecture)
        self.dense1 = nn.Linear(config['lstm_hidden_size'], config['dense_units'])  # 128->64
        self.output_layer = nn.Linear(config['dense_units'], self.output_features)  # 64->2
        self.sigmoid = nn.Sigmoid()  # Ensure output is in [0, 1] range
        
    def _summarize_lstm_sequence(self, lstm_output):
        """
        Summarize LSTM sequence output using attention mechanism
        """
        # Vectorized attention computation
        projected = self.attention_projection(lstm_output)  # (batch, seq, hidden//2)
        attention_weights = self.attention_tanh(projected)  # (batch, seq, hidden//2)
        attention_scores = self.attention_output(attention_weights)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq, 1)
        
        # Weighted sum
        summarized = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden)
        return summarized
    
    def forward(self, x):
        """Forward pass for v5.12.11 architecture"""
        batch_size, seq_len, features = x.size()
        
        # Reshape for CNN: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN processing
        x = self.cnn(x)
        x = self.cnn_bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention-based sequence summarization
        sequence_embedding = self._summarize_lstm_sequence(lstm_out)
        
        # Dense layers
        x = self.dense1(sequence_embedding)
        x = self.relu(x)
        output = self.output_layer(x)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        output = self.sigmoid(output)
        
        return output


class RealtimeVisualizer:
    """v5.12.11 True real-time EMG-to-angle prediction with continuous inference."""

    def __init__(self, serial_port: str = '/dev/cu.usbmodem178685801') -> None:
        # Canvas settings
        self.frame_width = 900
        self.frame_height = 600

        # Drawing configuration
        self.pred_color = (255, 100, 180)  # Pink/magenta
        self.joint_color = (255, 255, 255)  # White
        self.thickness_base = 4
        self.thickness_upper = 3

        # Segment length ratios relative to the base (MCP→PIP) segment
        self.pip_to_dip_ratio: float = 0.75
        self.dip_to_tip_ratio: float = 0.70
        
        # Display angles (updated by model inference)
        self.display_pred_pip_deg: float = 0.0
        self.display_pred_dip_deg: float = 0.0
        
        # Serial communication
        self.serial_port = serial_port
        self.baud_rate = 115200
        self.serial_connection = None
        self.emg_columns = ['A11', 'D11', 'D10', 'D9', 'D8', 'D7']
        
        # Live data collection
        self.latest_emg_data = {
            'A11': 0,  # Approximation coefficient (0-15.625Hz)
            'D11': 0,  # Detail coefficient (15.625-31.25Hz)
            'D10': 0,  # Detail coefficient (31.25-62.5Hz)
            'D9': 0,   # Detail coefficient (62.5-125Hz)
            'D8': 0,   # Detail coefficient (125-250Hz)
            'D7': 0    # Detail coefficient (250-500Hz)
        }
        self.emg_thread = None
        self.emg_running = False
        self.emg_data_lock = threading.Lock()
        
        # Initialization data (first 512 samples from interpolated_14.csv)
        self.init_df: Optional[pd.DataFrame] = None
        self.init_loaded: bool = False
        
        # Model loading and inference
        self.model: Optional[HybridCNNLSTM] = None
        self.scalers: Optional[dict] = None
        self.model_loaded: bool = False
        self.sequence_length: int = 512
        
        # Real-time buffer system (Producer-Consumer Pattern)
        self.buffer_size = 512
        self.buffer_lock = threading.Lock()
        
        # Rolling buffer for model inputs (normalized EMG deltas + scaled angles)
        self.model_input_buffer = np.zeros((self.buffer_size, 8), dtype=np.float32)
        self.buffer_ready = False  # True when buffer has 512 samples
        
        
        # Initialization state
        self.initialization_complete = False
        self.init_progress = 0
        self.init_samples_loaded = 0
        
        # EMG data tracking
        self.prev_raw_emg_data = None
        
        # Inference system
        self.prediction_result = None
        self.prediction_lock = threading.Lock()
        self.inference_thread = None
        self.running = False
        self.inference_in_progress = False
        self.last_inference_sample = -1  # Track which sample was last processed
        
        # Performance tracking
        self.buffer_update_count = 0
        self.inference_count = 0
        self.last_inference_time = 0.0
        self.inference_times = deque(maxlen=30)
        self.current_inference_rate = 0.0
        self.inference_intervals = deque(maxlen=10)  # Track intervals between inferences
        self.last_inference_completion = 0.0
        
        # Rate calculation with time window
        self.rate_calculation_start = 0.0
        self.rate_calculation_count = 0
        
        # Rolling average smoothing buffer
        self.smoothing_buffer_size: int = 10  # Will be set by user input
        self.prediction_smoothing_buffer = []  # Store recent predictions for averaging
        
        # Timing (no longer needed for buffer updates)
        
        # Load initialization data and model
        self._load_initialization_data()
        self._load_model()
        
        # Initialize serial connection
        self._init_serial()
        
        # Prompt user for smoothing buffer size
        self._get_smoothing_buffer_size()
        
        # Initialize buffer with first 512 samples from interpolated_14.csv
        if self.init_loaded:
            self._initialize_buffer()

    def _initialize_buffer(self) -> None:
        """Initialize the rolling buffer with first 512 samples from interpolated_14.csv."""
        print("[INFO] Initializing 512-sample rolling buffer from interpolated_14.csv...")
        print("[INFO] This will take a moment to prepare the system...")
        
        # Load first 512 samples from initialization data
        for i in range(min(self.buffer_size, len(self.init_df))):
            row = self.init_df.iloc[i]
            
            # Extract EMG data and angles
            emg_data = row[self.emg_columns].values.astype(np.float32)
            angle1 = float(row['angle1'])
            angle2 = float(row['angle2'])
            
            
            # Calculate EMG deltas (for first sample, delta is 0)
            if i == 0:
                emg_deltas = np.zeros_like(emg_data)
            else:
                prev_emg = self.init_df.iloc[i-1][self.emg_columns].values.astype(np.float32)
                emg_deltas = emg_data - prev_emg
            
            # Normalize EMG deltas
            if self.scalers and 'emg_deltas' in self.scalers:
                emg_deltas = self._normalize_emg_deltas(emg_deltas)
            
            # Scale angles to 0-1 range
            angle1_scaled = angle1 / 180.0
            angle2_scaled = angle2 / 180.0
            
            # Store in model input buffer
            self.model_input_buffer[i] = np.concatenate([emg_deltas, [angle1_scaled, angle2_scaled]])
            
            # Update progress
            self.init_samples_loaded = i + 1
            self.init_progress = (i + 1) / self.buffer_size * 100
            
            # Show progress every 50 samples
            if (i + 1) % 50 == 0:
                print(f"[INFO] Initialization progress: {self.init_progress:.1f}% ({self.init_samples_loaded}/{self.buffer_size})")
        
        # Initialize previous raw EMG data for delta calculation
        self.prev_raw_emg_data = emg_data.copy()
        
        self.buffer_ready = True
        self.initialization_complete = True
        print(f"[INFO] Buffer initialization complete! ({self.buffer_size} samples loaded)")
        print(f"[INFO] Ready for live serial data collection")
        print(f"[INFO] You can now start moving your finger - the system will begin live inference")

    def _load_initialization_data(self) -> None:
        """Load interpolated_14.csv for initialization buffer (first 512 samples)."""
        init_path = "datasets/og/interpolated_14.csv"
        
        if not os.path.exists(init_path):
            print(f"[ERROR] Initialization data not found: {init_path}")
            print("[ERROR] Cannot initialize live visualizer without initialization data")
            self.init_loaded = False
            return
        
        try:
            self.init_df = pd.read_csv(init_path)
            self.init_loaded = True
            
            print(f"[INFO] Loaded initialization data: {init_path}")
            print(f"[INFO] Total samples: {len(self.init_df):,}")
            print(f"[INFO] Columns: {list(self.init_df.columns)}")
            
            # Check for required columns
            if 'angle1' not in self.init_df.columns or 'angle2' not in self.init_df.columns:
                print("[ERROR] Required angle columns (angle1, angle2) not found in initialization data")
                self.init_loaded = False
                return
                
        except Exception as e:
            print(f"[ERROR] Failed to load initialization data: {e}")
            self.init_loaded = False

    def _load_model(self) -> None:
        """Load v5.12.11 model and scalers from outputs folder."""
        # Look for model files
        model_path = "outputs/models/final_model_v5.12.11.pt"
        scalers_path = "outputs/models/scalers_v5.12.11.pkl"
        
        if not os.path.exists(model_path):
            print(f"[WARNING] Model file not found: {model_path}")
            print("[INFO] Falling back to oscillation mode")
            return
        
        if not os.path.exists(scalers_path):
            print(f"[WARNING] Scalers file not found: {scalers_path}")
            print("[INFO] Falling back to oscillation mode")
            return
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create model with v5.12.11 config
            config = {
                'input_features': 8,
                'emg_delta_channels': 6,
                'angle_channels': 2,
                'output_features': 2,
                'sequence_length': 512,
                'cnn_filters': 64,
                'cnn_kernel_size': 5,
                'pool_size': 4,
                'final_cnn_timesteps': 32,
                'lstm_hidden_size': 128,
                'lstm_num_layers': 1,
                'dense_units': 64
            }
            
            self.model = HybridCNNLSTM(config)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Load scalers
            with open(scalers_path, 'rb') as f:
                self.scalers = joblib.load(f)
            
            self.model_loaded = True
            print(f"[INFO] Loaded v5.12.11 model and scalers")
            print(f"[INFO] Model: {model_path}")
            print(f"[INFO] Scalers: {scalers_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model_loaded = False


    def _init_serial(self) -> None:
        """Initialize serial connection for live EMG data collection."""
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            print(f"[INFO] Serial connection established on {self.serial_port}")
        except serial.SerialException as e:
            print(f"[WARNING] Could not connect to serial port {self.serial_port}: {e}")
            print("[WARNING] Live data collection will not be available")
            self.serial_connection = None
    
    def parse_serial_packet(self):
        """
        Parse a complete serial packet with six EMG coefficients.
        
        Format: 0x07 [4-byte A11][4-byte D11][4-byte D10][4-byte D9][4-byte D8][4-byte D7] 0x18
        
        Returns:
            Dictionary with EMG coefficients if valid packet found, None otherwise
        """
        if not self.serial_connection:
            return None
        
        try:
            # Look for header
            header = self.serial_connection.read(1)
            if not header or header[0] != 0x07:
                return None
            
            # Read the rest of the packet (6 * 4 bytes data + 1 byte footer)
            packet_data = self.serial_connection.read(25)  # 24 bytes data + 1 byte footer
            if len(packet_data) != 25:
                return None
            
            # Combine header and data
            full_packet = header + packet_data
            
            # Decode the packet using the new format
            if len(full_packet) != 26:  # Header(1) + Data(24) + Footer(1)
                return None
            
            # Check header and footer
            if full_packet[0] != 0x07 or full_packet[-1] != 0x18:
                return None
            
            # Extract six 32-bit signed integers (big-endian)
            # Each coefficient is 4 bytes
            a11_value = struct.unpack('>i', full_packet[1:5])[0]    # A11: 0-15.625Hz
            d11_value = struct.unpack('>i', full_packet[5:9])[0]    # D11: 15.625-31.25Hz
            d10_value = struct.unpack('>i', full_packet[9:13])[0]   # D10: 31.25-62.5Hz
            d9_value = struct.unpack('>i', full_packet[13:17])[0]   # D9: 62.5-125Hz
            d8_value = struct.unpack('>i', full_packet[17:21])[0]   # D8: 125-250Hz
            d7_value = struct.unpack('>i', full_packet[21:25])[0]   # D7: 250-500Hz
            
            # Return dictionary with all coefficients
            return {
                'A11': a11_value,
                'D11': d11_value,
                'D10': d10_value,
                'D9': d9_value,
                'D8': d8_value,
                'D7': d7_value
            }
            
        except Exception as e:
            # If any error occurs, return None
            return None
    
    def emg_monitoring_thread(self):
        """
        Continuous EMG monitoring thread that processes data on arrival.
        This thread continuously reads EMG data and immediately updates the buffer.
        """
        while self.emg_running:
            if self.serial_connection:
                # Try to get EMG data
                emg_data = self.parse_serial_packet()
                if emg_data:
                    # Update latest EMG data with thread safety
                    with self.emg_data_lock:
                        self.latest_emg_data = emg_data.copy()
                    
                    # Immediately update buffer with new data
                    self._update_live_buffer()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.0001)  # 0.1ms delay for responsiveness
    
    def _update_live_buffer(self) -> bool:
        """Update the rolling buffer with live EMG data and previous predictions. Returns True if successful."""
        if not self.serial_connection or not self.initialization_complete:
            return False
        
        try:
            with self.buffer_lock:
                # Get latest EMG data
                with self.emg_data_lock:
                    emg_data = np.array([
                        self.latest_emg_data['A11'],
                        self.latest_emg_data['D11'],
                        self.latest_emg_data['D10'],
                        self.latest_emg_data['D9'],
                        self.latest_emg_data['D8'],
                        self.latest_emg_data['D7']
                    ], dtype=np.float32)
                
                # Calculate EMG deltas using previous raw EMG data
                # We need to track the previous raw EMG data, not the deltas
                if not hasattr(self, 'prev_raw_emg_data'):
                    self.prev_raw_emg_data = emg_data.copy()
                
                emg_deltas = emg_data - self.prev_raw_emg_data
                self.prev_raw_emg_data = emg_data.copy()
                
                # Normalize EMG deltas
                emg_deltas = self._normalize_emg_deltas(emg_deltas)
                
                # Get previous model predictions for angle inputs (autoregressive)
                if self.prediction_result is not None:
                    with self.prediction_lock:
                        if self.prediction_result is not None:
                            angle1_scaled = np.clip(self.prediction_result[0], 0.0, 1.0)
                            angle2_scaled = np.clip(self.prediction_result[1], 0.0, 1.0)
                        else:
                            # Fallback to straight finger position
                            angle1_scaled = 0.5  # 90 degrees
                            angle2_scaled = 0.5  # 90 degrees
                else:
                    # First prediction, use straight finger position
                    angle1_scaled = 0.5  # 90 degrees
                    angle2_scaled = 0.5  # 90 degrees
                
                # Create new sample
                new_sample = np.concatenate([emg_deltas, [angle1_scaled, angle2_scaled]])
                
                # Shift buffer and add new sample
                self.model_input_buffer[:-1] = self.model_input_buffer[1:]
                self.model_input_buffer[-1] = new_sample
                
                
                self.buffer_update_count += 1
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to update live buffer: {e}")
            return False

    def _inference_worker(self) -> None:
        """Consumer thread that runs inference as fast as possible on live data."""
        while self.running:
            try:
                if not self.running:
                    break
                
                # Check if we have data to process
                if not self.buffer_ready or self.inference_in_progress:
                    time.sleep(0.001)  # Very short wait if no data or already processing
                    continue
                
                # Mark inference as in progress
                self.inference_in_progress = True
                
                # Get current buffer state
                with self.buffer_lock:
                    current_buffer = self.model_input_buffer.copy()
                
                # Run inference
                if self.model_loaded and self.model is not None:
                    start_time = time.time()
                    
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(current_buffer).unsqueeze(0)  # (1, 512, 8)
                        prediction = self.model(input_tensor)[0].numpy()  # (2,)
                    
                    # Apply rolling average smoothing
                    smoothed_prediction = self._apply_smoothing(prediction)
                    
                    # Update prediction result
                    with self.prediction_lock:
                        self.prediction_result = smoothed_prediction.copy()
                    
                    # Update performance tracking
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.inference_count += 1
                    
                    # Calculate inference rate using time window approach
                    current_time = time.time()
                    
                    # Initialize rate calculation start time
                    if self.rate_calculation_start == 0.0:
                        self.rate_calculation_start = current_time
                        self.rate_calculation_count = 0
                    
                    self.rate_calculation_count += 1
                    
                    # Calculate rate every 1 second
                    time_elapsed = current_time - self.rate_calculation_start
                    if time_elapsed >= 1.0:  # 1 second window
                        self.current_inference_rate = self.rate_calculation_count / time_elapsed
                        # Reset for next calculation
                        self.rate_calculation_start = current_time
                        self.rate_calculation_count = 0
                    
                    self.last_inference_completion = current_time
                    self.last_inference_time = current_time
                
                # Mark inference as complete
                self.inference_in_progress = False
                
            except Exception as e:
                print(f"[ERROR] Inference worker error: {e}")
                self.inference_in_progress = False
                time.sleep(0.001)  # Very brief pause on error

    def _start_inference_thread(self) -> None:
        """Start the inference worker thread."""
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.running = True
            self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self.inference_thread.start()
            print("[INFO] Inference thread started")
    
    def _start_emg_thread(self) -> None:
        """Start the EMG monitoring thread."""
        if self.emg_thread is None or not self.emg_thread.is_alive():
            self.emg_running = True
            self.emg_thread = threading.Thread(target=self.emg_monitoring_thread, daemon=True)
            self.emg_thread.start()
            print("[INFO] EMG monitoring thread started")

    def _stop_inference_thread(self) -> None:
        """Stop the inference worker thread."""
        self.running = False
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        print("[INFO] Inference thread stopped")
    
    def _stop_emg_thread(self) -> None:
        """Stop the EMG monitoring thread."""
        self.emg_running = False
        if self.emg_thread and self.emg_thread.is_alive():
            self.emg_thread.join(timeout=1.0)
        print("[INFO] EMG monitoring thread stopped")

    def _get_smoothing_buffer_size(self) -> None:
        """Prompt user for smoothing buffer size during initialization."""
        print("\n" + "="*60)
        print("🎛️  SMOOTHING BUFFER CONFIGURATION")
        print("="*60)
        print("Configure rolling average smoothing for model predictions:")
        print("- Larger buffer = smoother predictions (less noise)")
        print("- Smaller buffer = more responsive predictions")
        print("- Range: 1-50 samples")
        print()
        
        while True:
            try:
                buffer_input = input(f"Enter smoothing buffer size (default: 10): ").strip()
                
                if not buffer_input:  # Empty input, use default
                    self.smoothing_buffer_size = 10
                    print("✅ Using default smoothing buffer size: 10")
                    break
                
                buffer_size = int(buffer_input)
                
                if 1 <= buffer_size <= 50:
                    self.smoothing_buffer_size = buffer_size
                    print(f"✅ Smoothing buffer size set to: {self.smoothing_buffer_size}")
                    break
                else:
                    print("❌ Buffer size must be between 1 and 50. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n✅ Using default smoothing buffer size: 10")
                self.smoothing_buffer_size = 10
                break
        
        print("="*60)

    def _apply_smoothing(self, prediction: np.ndarray) -> np.ndarray:
        """Apply rolling average smoothing to model predictions."""
        # Add current prediction to buffer
        self.prediction_smoothing_buffer.append(prediction.copy())
        
        # Keep only the last smoothing_buffer_size predictions
        if len(self.prediction_smoothing_buffer) > self.smoothing_buffer_size:
            self.prediction_smoothing_buffer.pop(0)
        
        # Calculate rolling average
        if len(self.prediction_smoothing_buffer) > 0:
            smoothed_prediction = np.mean(self.prediction_smoothing_buffer, axis=0)
            return smoothed_prediction
        else:
            return prediction

    def _compute_emg_deltas(self, emg_sequence: np.ndarray) -> np.ndarray:
        """Compute EMG deltas using v5.12.11 method."""
        # Compute deltas using np.diff with prepend for same length
        emg_deltas = np.diff(emg_sequence, axis=0, prepend=emg_sequence[0:1])
        return emg_deltas.astype(np.float32)

    def _normalize_emg_deltas(self, emg_deltas: np.ndarray) -> np.ndarray:
        """Normalize EMG deltas using v5.12.11 scalers."""
        if self.scalers is None or 'emg_deltas' not in self.scalers:
            return emg_deltas
        
        try:
            normalized = emg_deltas.copy()
            emg_scalers = self.scalers['emg_deltas']
            
            # Handle both 1D (single sample) and 2D (sequence) arrays
            if emg_deltas.ndim == 1:
                # 1D array: single sample (6 EMG channels)
                for i, channel_key in enumerate(sorted(emg_scalers.keys())):
                    if i < len(emg_deltas):
                        stats = emg_scalers[channel_key]
                        mean_val = stats['mean']
                        std_val = stats['std']
                        normalized[i] = (emg_deltas[i] - mean_val) / (std_val + 1e-8)
            else:
                # 2D array: sequence of samples (time, 6 EMG channels)
                for i, channel_key in enumerate(sorted(emg_scalers.keys())):
                    if i < emg_deltas.shape[1]:
                        stats = emg_scalers[channel_key]
                        mean_val = stats['mean']
                        std_val = stats['std']
                        normalized[:, i] = (emg_deltas[:, i] - mean_val) / (std_val + 1e-8)
            
            return normalized
            
        except Exception as e:
            print(f"[ERROR] EMG delta normalization failed: {e}")
            return emg_deltas

    def _prepare_model_input(self, emg_sequence: np.ndarray, angle_sequence: np.ndarray) -> np.ndarray:
        """Prepare model input with EMG deltas and angles."""
        # Compute EMG deltas
        emg_deltas = self._compute_emg_deltas(emg_sequence)
        
        # Normalize EMG deltas
        normalized_emg_deltas = self._normalize_emg_deltas(emg_deltas)
        
        # Combine EMG deltas with angles
        model_input = np.concatenate([normalized_emg_deltas, angle_sequence], axis=1)
        
        return model_input

    def _run_model_inference(self, model_input: np.ndarray) -> Optional[np.ndarray]:
        """Run model inference on prepared input."""
        if not self.model_loaded or self.model is None:
            return None
        
        try:
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(model_input).unsqueeze(0)  # (1, 512, 8)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)[0].numpy()  # (2,)
            
            return prediction
            
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}")
            return None

    def _apply_smoothing(self, prediction: np.ndarray) -> np.ndarray:
        """Apply rolling average smoothing to model predictions."""
        # Add current prediction to buffer
        self.prediction_smoothing_buffer.append(prediction.copy())
        
        # Keep only the last smoothing_buffer_size predictions
        if len(self.prediction_smoothing_buffer) > self.smoothing_buffer_size:
            self.prediction_smoothing_buffer.pop(0)
        
        # Calculate rolling average
        if len(self.prediction_smoothing_buffer) > 0:
            smoothed_prediction = np.mean(self.prediction_smoothing_buffer, axis=0)
            return smoothed_prediction
        else:
            return prediction

    def _update_inference_rate(self) -> None:
        """Update the current inference rate based on recent inference times."""
        current_time = time.time()
        
        if self.last_inference_time > 0:
            # Calculate time since last inference
            time_diff = current_time - self.last_inference_time
            
            # Add to inference times history
            self.inference_times.append(time_diff)
            
            # Keep only the last max_inference_history times
            if len(self.inference_times) > self.max_inference_history:
                self.inference_times.pop(0)
            
            # Calculate average inference rate (inferences per second)
            if len(self.inference_times) > 1:
                avg_interval = np.mean(self.inference_times)
                self.current_inference_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        self.last_inference_time = current_time

    def _update_display_angles_live(self) -> None:
        """Update display angles from model predictions."""
        # Update predicted angles from model inference
        if self.prediction_result is not None:
            with self.prediction_lock:
                if self.prediction_result is not None:
                    # Convert from 0-1 range to degrees
                    pred_angle1 = np.clip(self.prediction_result[0], 0.0, 1.0) * 180.0
                    pred_angle2 = np.clip(self.prediction_result[1], 0.0, 1.0) * 180.0
                    
                    # Convert to display angles (bend angles)
                    self.display_pred_dip_deg = 180.0 - pred_angle1
                    self.display_pred_pip_deg = 180.0 - pred_angle2

    def synthesize_base(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Create a vertical base (MCP→PIP) centered-right on the canvas."""
        w, h = self.frame_width, self.frame_height
        base_len = int(min(w, h) * 0.22)
        center = (int(w * 0.52), int(h * 0.68))
        mcp_xy = (center[0], center[1] + base_len // 2)
        pip_xy = (center[0], center[1] - base_len // 2)
        return mcp_xy, pip_xy

    def compute_finger_points(
        self,
        base_start_xy: Tuple[int, int],
        base_end_xy: Tuple[int, int],
        pip_bend_deg: float,
        dip_bend_deg: float,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Return MCP, PIP, DIP, TIP in pixel coordinates."""
        mcp_xy = np.array(base_start_xy, dtype=float)
        pip_xy = np.array(base_end_xy, dtype=float)

        base_vec = pip_xy - mcp_xy
        base_len = float(np.linalg.norm(base_vec)) or 1.0
        base_dir = self.normalize(base_vec)

        pip_to_dip_len = base_len * self.pip_to_dip_ratio
        dip_to_tip_len = base_len * self.dip_to_tip_ratio

        pip_dir = self.rotate_vector(base_dir, pip_bend_deg)
        dip_xy = pip_xy + pip_dir * pip_to_dip_len

        dip_dir = self.rotate_vector(pip_dir, dip_bend_deg)
        tip_xy = dip_xy + dip_dir * dip_to_tip_len

        return (
            (int(mcp_xy[0]), int(mcp_xy[1])),
            (int(pip_xy[0]), int(pip_xy[1])),
            (int(dip_xy[0]), int(dip_xy[1])),
            (int(tip_xy[0]), int(tip_xy[1])),
        )

    @staticmethod
    def draw_finger(
        frame: np.ndarray,
        joints_xy: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness_base: int,
        thickness_upper: int,
        joint_color: Tuple[int, int, int],
    ) -> None:
        """Draw a stick finger with joints."""
        mcp, pip, dip, tip = joints_xy
        cv2.line(frame, mcp, pip, color, thickness_base, cv2.LINE_AA)
        cv2.line(frame, pip, dip, color, thickness_upper, cv2.LINE_AA)
        cv2.line(frame, dip, tip, color, thickness_upper, cv2.LINE_AA)
        for pt in (mcp, pip, dip, tip):
            cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)


    def render_frame(self, t: float) -> np.ndarray:
        """Render a frame with current joint angles."""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Update angles from live data
        self._update_display_angles_live()
        
        # Get base finger position
        mcp_xy, pip_xy = self.synthesize_base()
        
        # Compute joints and draw predicted finger
        pred_joints = self.compute_finger_points(mcp_xy, pip_xy, self.display_pred_pip_deg, self.display_pred_dip_deg)

        self.draw_finger(frame, pred_joints, color=self.pred_color,
                         thickness_base=self.thickness_base, thickness_upper=self.thickness_upper,
                         joint_color=self.joint_color)

        # UI labels
        cv2.putText(frame, "Live EMG Visualizer (v5.12.11)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (200, 200, 0), 2, cv2.LINE_AA)
        
        if self.init_loaded:
            if self.model_loaded and self.initialization_complete:
                cv2.putText(frame, "Live Serial Data Collection", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 0), 2, cv2.LINE_AA)
                # Serial connection status
                if self.serial_connection:
                    serial_status = "Serial: CONNECTED"
                    serial_color = (0, 255, 0)  # Green
                else:
                    serial_status = "Serial: DISCONNECTED"
                    serial_color = (255, 0, 0)  # Red
                cv2.putText(frame, serial_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, serial_color, 1, cv2.LINE_AA)
                legend_text = "Legend: PRED=magenta  |  'q'=quit, 'r'=reset, 's'=clear buffer"
            elif self.model_loaded and not self.initialization_complete:
                cv2.putText(frame, "Initializing System...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 165, 0), 2, cv2.LINE_AA)
                progress = f"Progress: {self.init_progress:.1f}% ({self.init_samples_loaded}/{self.buffer_size})"
                cv2.putText(frame, progress, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
                legend_text = "Loading ground truth buffer from interpolated_14.csv..."
            else:
                cv2.putText(frame, "No Model Loaded", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2, cv2.LINE_AA)
                legend_text = "Please ensure model files are available"
        else:
            cv2.putText(frame, "No Initialization Data", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2, cv2.LINE_AA)
            legend_text = "Please ensure interpolated_14.csv is available"
        
        cv2.putText(frame, legend_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
        
        # Angle displays (matching hand_tracking.py joint mapping)
        cv2.putText(frame, f"PIP: {self.display_pred_pip_deg:+.1f}°", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.pred_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"DIP: {self.display_pred_dip_deg:+.1f}°", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.pred_color, 1, cv2.LINE_AA)
        
        # Performance info
        if self.buffer_ready:
            buffer_text = f"Buffer Updates: {self.buffer_update_count:,}"
            cv2.putText(frame, buffer_text, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            
            inference_text = f"Inferences: {self.inference_count:,}"
            cv2.putText(frame, inference_text, (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            
            # Show smoothing buffer status
            smoothing_text = f"Smoothing: {len(self.prediction_smoothing_buffer)}/{self.smoothing_buffer_size}"
            cv2.putText(frame, smoothing_text, (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            
            # Show inference status
            if self.inference_in_progress:
                status_text = "Inference: RUNNING"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "Inference: IDLE"
                status_color = (100, 100, 100)  # Gray
            
            cv2.putText(frame, status_text, (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
        
        # Inference rate display (top-right corner)
        if self.model_loaded and self.buffer_ready and self.current_inference_rate > 0:
            # Cap the displayed rate at a reasonable maximum
            display_rate = min(self.current_inference_rate, 50.0)  # Cap at 50 Hz
            inference_text = f"Inference Rate: {display_rate:.1f} Hz"
            # Get text size for positioning
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(inference_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position in top-right corner with padding
            x_pos = self.frame_width - text_width - 20
            y_pos = 40
            
            # Draw background rectangle for better visibility
            cv2.rectangle(frame, (x_pos - 10, y_pos - text_height - 10), 
                         (x_pos + text_width + 10, y_pos + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (x_pos - 10, y_pos - text_height - 10), 
                         (x_pos + text_width + 10, y_pos + 5), (100, 100, 100), 2)
            
            # Draw text
            cv2.putText(frame, inference_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        
        return frame


    # ------------------------------ Geometry helpers ------------------------------
    @staticmethod
    def rotate_vector(vec: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate a 2D vector by the given angle in degrees."""
        angle_radians = math.radians(angle_degrees)
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return rot @ vec

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        n = float(np.linalg.norm(vec))
        if n == 0.0:
            return vec
        return vec / n

    def run(self) -> None:
        """Main live visualization loop with serial data collection."""
        win_name = "Live EMG Visualizer (v5.12.11)"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        
        if not self.init_loaded:
            print("[ERROR] No initialization data loaded. Cannot run live visualizer.")
            return
        
        if not self.model_loaded:
            print("[ERROR] No model loaded. Cannot run live visualizer.")
            return
        
        if not self.buffer_ready:
            print("[ERROR] Buffer not ready. Cannot run live visualizer.")
            return
        
        # Start EMG thread first
        self._start_emg_thread()
        
        t0 = time.time()
        
        print("[INFO] Starting live EMG visualization...")
        print("[INFO] System is initializing - please wait...")
        print("[INFO] EMG collection on data arrival, inference runs as fast as possible")
        print("[INFO] Controls: 'q'=quit, 'r'=reset, 's'=clear buffer")
        
        # Wait for initialization to complete
        while not self.initialization_complete:
            current_time = time.time()
            t = current_time - t0
            
            # Render frame during initialization
            frame = self.render_frame(t)
            cv2.imshow(win_name, frame)
            
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                return
            
            time.sleep(0.01)  # Small delay during initialization
        
        # Start inference thread after initialization
        self._start_inference_thread()
        print("[INFO] Initialization complete! Starting live inference...")
        print("[INFO] You can now start moving your finger")
        
        try:
            while True:
                current_time = time.time()
                t = current_time - t0
                
                # Render frame (buffer updates happen automatically in EMG thread)
                frame = self.render_frame(t)
                
                # Display frame
                cv2.imshow(win_name, frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Reset
                    self.buffer_update_count = 0
                    self.inference_count = 0
                    self.prediction_smoothing_buffer.clear()
                    print("[INFO] Reset complete")
                elif key == ord('s'):  # Clear smoothing buffer
                    self.prediction_smoothing_buffer.clear()
                    print("[INFO] Smoothing buffer cleared")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            # Stop threads
            self._stop_inference_thread()
            self._stop_emg_thread()
            
            # Close serial connection
            if self.serial_connection:
                self.serial_connection.close()
                print("[INFO] Serial connection closed")
            
            cv2.destroyAllWindows()
            print("[INFO] Live EMG visualizer closed")


if __name__ == "__main__":
    # Parse command line arguments
    default_port = '/dev/cu.usbmodem178685801'
    
    if len(sys.argv) > 1:
        serial_port = sys.argv[1]
    else:
        serial_port = default_port
    
    print("[INFO] Starting Live EMG Visualizer (v5.12.11)")
    print(f"[INFO] Serial port: {serial_port}")
    print("[INFO] Features:")
    print("  - Live serial EMG data collection at 1kHz")
    print("  - 512-sample rolling buffer with autoregressive prediction")
    print("  - Model inference runs as fast as possible (no artificial delays)")
    print("  - Initialization from interpolated_14.csv (straight finger)")
    print("[INFO] Controls:")
    print("  'q' or ESC - Quit")
    print("  'r' - Reset")
    print("  's' - Clear smoothing buffer")
    
    visualizer = RealtimeVisualizer(serial_port)
    visualizer.run()
