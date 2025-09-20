
# Model Structure for Deployment - v5.12.11
# ==========================================

import torch
import torch.nn as nn
import numpy as np
import pickle
import joblib
from pathlib import Path

class HybridCNNLSTM(nn.Module):
    """
    EMG-Delta optimized CNN-LSTM model for angle prediction - v5.12.11
    Designed for 1kHz sampling, 512ms context, 6 EMG deltas + 2 angles
    """
    def __init__(self, config):
        super(HybridCNNLSTM, self).__init__()
        
        self.config = config
        self.input_features = config['input_features']
        self.emg_channels = config['emg_channels']
        self.angle_channels = config['angle_channels']
        self.output_features = config['output_features']
        self.sequence_length = config['sequence_length']
        
        # EMG Delta Processing Branch
        self.emg_burst_conv = nn.Conv1d(
            in_channels=self.emg_channels,
            out_channels=config['emg_filters'],
            kernel_size=config['emg_burst_kernel'],
            padding=config['emg_burst_kernel']//2,
            bias=False
        )
        self.emg_burst_bn = nn.BatchNorm1d(config['emg_filters'])
        self.emg_pool1 = nn.MaxPool1d(kernel_size=config['pool_size'])
        
        # Angle Processing Branch
        self.angle_conv = nn.Conv1d(
            in_channels=self.angle_channels,
            out_channels=config['angle_filters'],
            kernel_size=config['angle_kernel'],
            padding=config['angle_kernel']//2,
            bias=False
        )
        self.angle_bn = nn.BatchNorm1d(config['angle_filters'])
        self.angle_pool1 = nn.MaxPool1d(kernel_size=config['pool_size'])
        
        # Combined Feature Integration
        combined_input_channels = config['emg_filters'] + config['angle_filters']
        self.pattern_conv = nn.Conv1d(
            in_channels=combined_input_channels,
            out_channels=config['combined_filters'],
            kernel_size=config['emg_pattern_kernel'],
            padding=config['emg_pattern_kernel']//2,
            bias=False
        )
        self.pattern_bn = nn.BatchNorm1d(config['combined_filters'])
        self.pattern_pool = nn.MaxPool1d(kernel_size=config['pool_size'])
        
        # LSTM Sequence Processing
        self.cnn_output_size = config['combined_filters']
        self.final_timesteps = config['final_cnn_timesteps']
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            bidirectional=config['lstm_bidirectional']
        )
        
        # Dense Layers
        self.dense1 = nn.Linear(config['lstm_hidden_size'], config['dense_layers'][0])
        self.output_layer = nn.Linear(config['dense_layers'][0], self.output_features)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
    
    def _calculate_cnn_output_size(self):
        """Calculate the output size after CNN layers"""
        dummy_input = torch.randn(1, self.input_features, self.sequence_length)
        
        with torch.no_grad():
            x = dummy_input
            for layer in self.cnn_layers:
                if isinstance(layer, (nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d, nn.ReLU, nn.GELU, nn.SiLU, nn.Dropout)):
                    x = layer(x)
            return x.size(1)
    
    def forward(self, x):
        """EMG-Delta optimized forward pass"""
        batch_size, seq_len, features = x.size()
        
        # Reshape for CNN: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # Split EMG deltas and angles
        emg_deltas = x[:, :self.emg_channels, :]
        angles = x[:, self.emg_channels:, :]
        
        # EMG Delta Processing Branch
        emg_features = self.emg_burst_conv(emg_deltas)
        emg_features = self.emg_burst_bn(emg_features)
        emg_features = self.relu(emg_features)
        emg_features = self.emg_pool1(emg_features)
        
        # Angle Processing Branch
        angle_features = self.angle_conv(angles)
        angle_features = self.angle_bn(angle_features)
        angle_features = self.relu(angle_features)
        angle_features = self.angle_pool1(angle_features)
        
        # Combined Feature Integration
        combined_features = torch.cat([emg_features, angle_features], dim=1)
        pattern_features = self.pattern_conv(combined_features)
        pattern_features = self.pattern_bn(pattern_features)
        pattern_features = self.relu(pattern_features)
        pattern_features = self.pattern_pool(pattern_features)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        lstm_input = pattern_features.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        sequence_embedding = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.dense1(sequence_embedding)
        x = self.relu(x)
        output = self.output_layer(x)
        
        return output

def load_model_for_inference(model_path, scaler_path=None, device='cpu'):
    """
    Load trained model for inference
    
    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scalers
        device: Device to load model on
        
    Returns:
        Loaded model and scalers
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    config = checkpoint['hyperparameters']['model']
    
    # Initialize model
    model = HybridCNNLSTM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load scalers if provided
    scalers = None
    if scaler_path and Path(scaler_path).exists():
        scalers = joblib.load(scaler_path)
    
    return model, scalers, config

def predict_angles(model, input_sequence, scalers=None, device='cpu'):
    """
    Predict angles from input sequence
    
    Args:
        model: Trained model
        input_sequence: Input sequence (sequence_length, features)
        scalers: Data scalers
        device: Device to run inference on
        
    Returns:
        Predicted angles (angle1, angle2)
    """
    model.eval()
    
    # Ensure input is the right shape
    if input_sequence.ndim == 2:
        input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
    
    # Scale input if scalers provided
    if scalers and 'combined' in scalers:
        # Reshape for scaling
        original_shape = input_sequence.shape
        seq_reshaped = input_sequence.reshape(-1, input_sequence.shape[-1])
        seq_scaled = scalers['combined'].transform(seq_reshaped.numpy())
        input_sequence = torch.FloatTensor(seq_scaled).reshape(original_shape)
    
    # Move to device
    input_sequence = input_sequence.to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(input_sequence)
    
    # Convert back to numpy
    predictions = predictions.cpu().numpy()
    
    # Inverse scale if scalers provided
    if scalers and 'combined' in scalers:
        # Create dummy array for inverse transform
        dummy_features = np.zeros((predictions.shape[0], scalers['combined'].n_features))
        dummy_features[:, :2] = predictions  # Only angle predictions
        
        # Inverse transform
        predictions_original = scalers['combined'].inverse_transform(dummy_features)[:, :2]
        return predictions_original
    
    return predictions

# Example usage:
# model, scalers, config = load_model_for_inference('model_v5.12.11.pt', 'scalers_v5.12.11.pkl')
# predictions = predict_angles(model, input_sequence, scalers)
