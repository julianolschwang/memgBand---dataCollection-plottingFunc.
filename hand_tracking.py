#!/usr/bin/env python3
"""
Hand tracking with MediaPipe and Serial data collection.
Integrates MediaPipe hand detection with camera feed and Serial EMG data.

FEATURES:
- Real-time hand tracking with MediaPipe
- Serial EMG data collection from device
- Synchronized angle and EMG data recording
- CSV file output for analysis

DATA FLOW:
- Script reads EMG data from: /dev/cu.usbmodem178685801
- Hand tracking runs simultaneously with data collection
- Data is saved to CSV files for analysis
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import csv
import os
import struct
import threading

class HandTracker:
    def __init__(self, serial_port='/dev/cu.usbmodem178685801', baud_rate=115200):
        # Initialize MediaPipe with optimized settings for 30 FPS
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lowered for faster processing
            min_tracking_confidence=0.3,    # Lowered for faster processing
            model_complexity=0              # Use fastest model (0=light, 1=full)
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Index finger landmarks (TIP, DIP, PIP, MCP)
        self.index_finger_landmarks = [8, 7, 6, 5]
        
        # Serial communication
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.recording = False
        self.serial_connection = None
        

        
        # CSV file handling
        self.csv_file = None
        self.csv_writer = None
        self.datasets_dir = "datasets"
        
        # EMG-driven data collection
        self.new_angle_flag = False
        self.latest_angles = None
        self.latest_timestamp = None
        
        # Data collection limits
        self.sample_count = 0
        self.max_samples = 30000  # Stop after 30,000 samples (30 seconds at 1kHz)
        
        # Threading for continuous EMG monitoring
        self.emg_thread = None
        self.emg_running = False
        
        # EMG data storage for the six coefficients
        self.latest_emg_data = {
            'A11': 0,  # Approximation coefficient (0-15.625Hz)
            'D11': 0,  # Detail coefficient (15.625-31.25Hz)
            'D10': 0,  # Detail coefficient (31.25-62.5Hz)
            'D9': 0,   # Detail coefficient (62.5-125Hz)
            'D8': 0,   # Detail coefficient (125-250Hz)
            'D7': 0    # Detail coefficient (250-500Hz)
        }
        
        # Hand tracking validation
        self.hand_lost_frames = 0
        self.max_lost_frames = 10  # Cancel recording after 10 consecutive frames without hand detection
        
        # Logging control (no delay - start immediately)
        self.logging_started = True        # Start logging immediately
        self.recording_start_time = None   # Time when recording started
        
        # Ensure datasets directory exists
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)
        
        # Initialize serial connection
        self.init_serial()
        
        # Plotting functionality removed - simple camera and data collection only
    
    def get_recording_status(self):
        """
        Get current recording status information.
        
        Returns:
            Dictionary with recording status information
        """
        if not self.recording:
            return None
        
        elapsed_time = time.time() - self.recording_start_time if self.recording_start_time else 0
        progress = self.sample_count / self.max_samples if self.max_samples > 0 else 0
        
        return {
            'samples_collected': self.sample_count,
            'samples_target': self.max_samples,
            'time_elapsed': elapsed_time,
            'progress': progress
        }
    
    def init_serial(self):
        
        """Initialize serial connection."""
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0.1
            )
            print(f"Serial connection established on {self.serial_port}")
        except serial.SerialException as e:
            print(f"Warning: Could not connect to serial port {self.serial_port}: {e}")
            print("Data collection will continue without serial data.")
            self.serial_connection = None
    
            # All plotting methods removed - simple functionality only
    
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
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points.
        
        Args:
            point1: First point (x, y)
            point2: Middle point (vertex of angle)
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Ensure value is within valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle) * 180 / np.pi
        
        return angle
    
    def get_index_finger_angles(self, hand_landmarks, frame_width, frame_height):
        """
        Calculate angles between index finger joints.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            List of angles between consecutive joints
        """
        angles = []
        index_points = []
        
        # Extract index finger landmark positions and convert to pixel coordinates
        for landmark_id in self.index_finger_landmarks:
            landmark = hand_landmarks.landmark[landmark_id]
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            index_points.append((x, y))
        
        # Calculate angles between consecutive joints
        for i in range(len(index_points) - 2):
            angle = self.calculate_angle(
                index_points[i], 
                index_points[i + 1], 
                index_points[i + 2]
            )
            angles.append(angle)
        
        return angles
    
    def draw_index_finger_only(self, frame, hand_landmarks, angles):
        """
        Draw only index finger landmarks, connections, and angle information.
        
        Args:
            frame: Input frame
            hand_landmarks: MediaPipe hand landmarks
            angles: List of calculated angles
        """
        h, w, _ = frame.shape
        
        # Draw index finger landmarks with different colors for each joint
        landmark_colors = [
            (0, 255, 255),  # TIP - Yellow
            (0, 255, 0),    # DIP - Green  
            (255, 0, 0),    # PIP - Blue
            (0, 0, 255)     # MCP - Red
        ]
        
        for i, landmark_id in enumerate(self.index_finger_landmarks):
            landmark = hand_landmarks.landmark[landmark_id]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 6, landmark_colors[i], -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)  # White border
        
        # Draw connections between index finger joints
        for i in range(len(self.index_finger_landmarks) - 1):
            landmark1 = hand_landmarks.landmark[self.index_finger_landmarks[i]]
            landmark2 = hand_landmarks.landmark[self.index_finger_landmarks[i + 1]]
            
            x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
            x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)  # White connections
        
        # Draw angle annotations with better positioning
        joint_names = ["TIP-DIP", "DIP-PIP"]
        for i, angle in enumerate(angles):
            if i < len(self.index_finger_landmarks) - 2:
                landmark = hand_landmarks.landmark[self.index_finger_landmarks[i + 1]]
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Position text to avoid overlap
                text_x = x + 15
                text_y = y - 10 - (i * 25)  # Offset each angle text
                
                # Draw angle text with background for better readability
                text = f"{joint_names[i]}: {angle:.1f}°"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2), 
                             (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def check_and_cancel_recording(self, frame, reason):
        """
        Check if recording should be cancelled due to hand loss.
        
        Args:
            frame: Current frame for displaying warning
            reason: Reason for potential cancellation
        """
        # Display warning on frame
        warning_text = f"WARNING: {reason}"
        cv2.putText(frame, warning_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        lost_count_text = f"Lost frames: {self.hand_lost_frames}/{self.max_lost_frames}"
        cv2.putText(frame, lost_count_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Cancel recording if too many frames lost
        if self.hand_lost_frames >= self.max_lost_frames:
            print(f"\n[WARNING] Recording cancelled: {reason}")
            print(f"[INFO] Hand tracking lost for {self.hand_lost_frames} consecutive frames")
            print(f"[INFO] Collected {self.sample_count:,} samples before cancellation")
            
            # Stop recording
            self.recording = False
            self.stop_csv_recording()
            
            # Reset counters
            self.hand_lost_frames = 0
            
            # Display cancellation message on frame
            cancel_text = "RECORDING CANCELLED - Hand Lost"
            cv2.putText(frame, cancel_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    
    def process_frame(self, frame):
        """
        Process a single frame for hand tracking and data collection.
        Optimized for 30 FPS performance.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with hand tracking
        """
        # Convert BGR to RGB for MediaPipe (optimized)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            # Hand detected - reset lost frame counter
            self.hand_lost_frames = 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate index finger angles
                angles = self.get_index_finger_angles(
                    hand_landmarks, frame.shape[1], frame.shape[0]
                )
                
                # Validate that we have valid angles
                if angles and len(angles) >= 2:
                    # Draw only index finger landmarks and connections
                    self.draw_index_finger_only(frame, hand_landmarks, angles)
                    
                    # Display angle information (optimized text rendering)
                    angle_text = f"Angles: {angles[0]:.0f}°, {angles[1]:.0f}°"
                    cv2.putText(frame, angle_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Update angle data and set flag for EMG-driven collection
                    self.latest_angles = angles
                    self.latest_timestamp = time.time()
                    self.new_angle_flag = True
                    
                    # Show recording status
                    if self.recording:
                        status_text = "RECORDING - Hand Tracked"
                        cv2.putText(frame, status_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Invalid angles - treat as hand lost
                    self.hand_lost_frames += 1
                    if self.recording:
                        self.check_and_cancel_recording(frame, "Invalid hand pose detected")
        else:
            # No hand detected
            self.hand_lost_frames += 1
            if self.recording:
                self.check_and_cancel_recording(frame, "Hand lost from frame")
            else:
                # Show instruction when not recording
                instruction_text = "Show your hand to start tracking"
                cv2.putText(frame, instruction_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Always show recording controls
        if not self.recording:
            control_text = "Press 'r' to start recording | Press 'q' to quit"
            cv2.putText(frame, control_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the hand tracking application with data collection."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Set camera properties for optimal 30 FPS performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG for faster encoding
        
        print("Hand Tracking with EMG-Driven Data Collection Started!")
        print("Show your hand to the camera.")
        print("Press 'r' in camera window to START data collection")
        print("Data collection will automatically stop after 30,000 samples (30 seconds)")
        print("Recording will CANCEL if hand is lost for more than 10 frames (~0.33s)")
        print("Logging starts immediately when recording begins")
        print("Press 'q' to quit")
        print("Data format: timestamp,angle1,angle2,A11,D11,D10,D9,D8,D7")
        print("EMG sampling includes six wavelet coefficients (A11 + D11-D7)")
        print("-" * 60)
        
        # Start EMG monitoring thread
        self.emg_running = True
        self.emg_thread = threading.Thread(target=self.emg_monitoring_thread, daemon=True)
        self.emg_thread.start()
        print("[INFO] EMG monitoring thread started")
        print("[INFO] Thread will continue running for multiple recording sessions")
        
        # Plotting functionality removed - simple camera and data collection only
        
        # Performance monitoring
        frame_count = 0
        start_time = time.time()
        target_frame_time = 1.0 / 30.0  # Target 30 FPS
        last_frame_time = time.time()
        
        try:
            while True:
                loop_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Process frame for hand tracking
                processed_frame = self.process_frame(frame)
                
                # EMG processing is now handled by separate thread
                
                # Calculate and display FPS with precise timing
                frame_count += 1
                current_time = time.time()
                
                if frame_count % 30 == 0:  # Every second at 30 FPS
                    elapsed_time = current_time - start_time
                    fps = 30 / elapsed_time
                    recording_status = "RECORDING" if self.recording else "IDLE"
                    frame_time = (current_time - last_frame_time) * 1000  # Convert to ms
                    
                    if self.recording:
                        progress_pct = (self.sample_count / self.max_samples) * 100 if self.max_samples > 0 else 0
                        sample_info = f", Samples: {self.sample_count:,}/{self.max_samples:,} ({progress_pct:.1f}%)"
                    else:
                        sample_info = ""
                    
                    print(f"[STATUS] FPS: {fps:.1f}, Frame Time: {frame_time:.1f}ms, Status: {recording_status}{sample_info}")
                    start_time = current_time
                    last_frame_time = current_time
                
                # Display the frame
                cv2.imshow('Hand Tracking', processed_frame)
                
                # Plot responsiveness removed - simple functionality only
                
                # Handle key presses with minimal delay
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not self.recording:
                    # Start recording (only if not already recording)
                    self.recording = True
                    self.sample_count = 0  # Reset sample counter
                    print("[INFO] Data collection STARTED")
                    print(f"[INFO] Will automatically stop after {self.max_samples:,} samples (30 seconds)")
                    print(f"[INFO] Logging starts immediately")
                    print(f"[DEBUG] Recording state set to: {self.recording}")
                    self.start_csv_recording()
                elif key == ord('r') and self.recording:
                    # If already recording, show status
                    progress_pct = (self.sample_count / self.max_samples) * 100 if self.max_samples > 0 else 0
                    print(f"[INFO] Already recording. Current samples: {self.sample_count:,}/{self.max_samples:,} ({progress_pct:.1f}%)")
                
                # Frame rate limiting to maintain 30 FPS
                loop_time = time.time() - loop_start_time
                if loop_time < target_frame_time:
                    time.sleep(target_frame_time - loop_time)
        
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        finally:
            # Stop EMG monitoring thread
            self.emg_running = False
            if self.emg_thread:
                self.emg_thread.join(timeout=1.0)
                print("[INFO] EMG monitoring thread stopped")
            
            # Plotting cleanup removed - simple functionality only
            
            # Stop CSV recording if active
            if self.recording:
                self.stop_csv_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
            if self.serial_connection:
                self.serial_connection.close()
            
            print("Hand tracking stopped.")

    def start_csv_recording(self):
        """Start recording data to a new CSV file."""
        # Find the next available file number
        file_number = 1
        while os.path.exists(os.path.join(self.datasets_dir, f"{file_number}.csv")):
            file_number += 1
        
        csv_path = os.path.join(self.datasets_dir, f"{file_number}.csv")
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header line with all six EMG coefficients
        header = ['timestamp', 'angle1', 'angle2', 'A11', 'D11', 'D10', 'D9', 'D8', 'D7']
        self.csv_writer.writerow(header)
        
        print(f"[INFO] Started recording to {csv_path}")
        print(f"[INFO] Recording format: timestamp,angle1,angle2,A11,D11,D10,D9,D8,D7")
        print(f"[INFO] Logging starts immediately")
        
        # Initialize recording
        self.recording_start_time = time.time()  # Set recording start time
        self.logging_started = True              # Start logging immediately
    
    def stop_csv_recording(self):
        """Stop recording and close the CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print("[INFO] Stopped recording and saved CSV file")
    
    def write_data_to_csv(self, timestamp, angles, emg_data):
        """Write a data row to the CSV file with all six EMG coefficients."""
        if self.csv_writer:
            # Format: timestamp,angle1,angle2,A11,D11,D10,D9,D8,D7
            if angles:
                row = [timestamp, angles[0], angles[1], emg_data['A11'], emg_data['D11'], emg_data['D10'], emg_data['D9'], emg_data['D8'], emg_data['D7']]
            else:
                # No angle data available, use zeros
                row = [timestamp, 0, 0, emg_data['A11'], emg_data['D11'], emg_data['D10'], emg_data['D9'], emg_data['D8'], emg_data['D7']]
            self.csv_writer.writerow(row)
    

    
    def emg_monitoring_thread(self):
        """
        Continuous EMG monitoring thread that runs at 1kHz.
        This thread continuously reads EMG data and handles pairing with angle data.
        """
        while self.emg_running:
            if self.recording:
                # Try to get EMG data if serial connection exists
                if self.serial_connection:
                    emg_data = self.parse_serial_packet()
                    if emg_data:
                        timestamp = time.time()
                        
                        # Store latest EMG data for potential future use
                        self.latest_emg_data = emg_data.copy()
                        
                        # Log data immediately (no delay)
                        if self.new_angle_flag and self.latest_angles:
                            # Pair EMG with latest angle data
                            self.write_data_to_csv(timestamp, self.latest_angles, emg_data)
                            self.new_angle_flag = False  # Reset flag after pairing
                            
                            # Print paired data with all coefficients
                            print(f"{timestamp:.6f},{self.latest_angles[0]:.2f},{self.latest_angles[1]:.2f},{emg_data['A11']},{emg_data['D11']},{emg_data['D10']},{emg_data['D9']},{emg_data['D8']},{emg_data['D7']}")
                        else:
                            # Save EMG data without angle pairing
                            self.write_data_to_csv(timestamp, None, emg_data)
                            
                            # Print EMG-only data with all coefficients
                            print(f"{timestamp:.6f},0,0,{emg_data['A11']},{emg_data['D11']},{emg_data['D10']},{emg_data['D9']},{emg_data['D8']},{emg_data['D7']}")
                        
                        # Increment sample counter
                        self.sample_count += 1
                        
                        # Check if we've reached the sample limit
                        if self.sample_count >= self.max_samples:
                            elapsed_time = time.time() - self.recording_start_time if self.recording_start_time else 0
                            print(f"[INFO] Reached {self.max_samples:,} samples ({elapsed_time:.1f}s). Stopping data collection.")
                            self.recording = False
                            self.stop_csv_recording()
                            print(f"[DEBUG] Recording stopped. Thread continues monitoring. Recording state: {self.recording}")
                            # Don't break - just continue monitoring for next recording session
            
            # Small delay to maintain ~1kHz sampling rate
            time.sleep(0.001)  # 1ms delay for 1kHz sampling

def main():
    """Main function."""
    # Default values
    default_port = '/dev/cu.usbmodem178685801'
    default_baud = 115200
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        serial_port = sys.argv[1]
    else:
        serial_port = default_port
        
    if len(sys.argv) > 2:
        try:
            baud_rate = int(sys.argv[2])
        except ValueError:
            print(f"Invalid baudrate: {sys.argv[2]}, using default {default_baud}")
            baud_rate = default_baud
    else:
        baud_rate = default_baud
    
    print("Hand Tracking with MediaPipe and Serial Data Collection")
    print("=" * 60)
    print(f"Serial port: {serial_port}")
    print(f"Baudrate: {baud_rate}")
    print("This will track your hand and collect synchronized angle/EMG data.")
    print()
    
    tracker = HandTracker(serial_port=serial_port, baud_rate=baud_rate)
    
    try:
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 