#!/usr/bin/env python3
"""
Real-time Wavelet Coefficients Plotter

This script receives binary data from the Teensy and plots all 6 wavelet coefficients
(A11, D11, D10, D9, D8, D7) in real-time on a scrolling plot.

Usage:
    python3 serial_plotter.py [port] [baudrate]
    
Example:
    python3 serial_plotter.py /dev/cu.usbmodem178685801 115200
"""

import sys
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import serial as pyserial

class WaveletPlotter:
    def __init__(self, port, baudrate=115200, max_points=1000):
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points
        
        # Data storage
        self.times = deque(maxlen=max_points)
        self.coefficients = {
            'A11': deque(maxlen=max_points),  # 0-15.625 Hz
            'D11': deque(maxlen=max_points),  # 15.625-31.25 Hz
            'D10': deque(maxlen=max_points),  # 31.25-62.5 Hz
            'D9': deque(maxlen=max_points),   # 62.5-125 Hz
            'D8': deque(maxlen=max_points),   # 125-250 Hz
            'D7': deque(maxlen=max_points)    # 250-500 Hz
        }
        
        # Threading
        self.serial_thread = None
        self.running = False
        self.data_lock = threading.Lock()
        
        # Plot setup with dark theme
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()
        
        # Serial connection
        self.serial_conn = None
        
    def setup_plot(self):
        """Initialize the plot with dark theme and styling."""
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Create lines for all 6 bands
        self.line_a11, = self.ax.plot([], [], 'cyan', linewidth=2.0, label='A11 (0-15.625Hz)')
        self.line_d11, = self.ax.plot([], [], 'red', linewidth=2.0, label='D11 (15.625-31.25Hz)')
        self.line_d10, = self.ax.plot([], [], 'green', linewidth=2.0, label='D10 (31.25-62.5Hz)')
        self.line_d9, = self.ax.plot([], [], 'yellow', linewidth=2.0, label='D9 (62.5-125Hz)')
        self.line_d8, = self.ax.plot([], [], 'magenta', linewidth=2.0, label='D8 (125-250Hz)')
        self.line_d7, = self.ax.plot([], [], 'orange', linewidth=2.0, label='D7 (250-500Hz)')
        
        # Configure plot
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(-2**31, 2**31)
        self.ax.set_xlabel('Sample Number', color='white')
        self.ax.set_ylabel('Coefficient Value', color='white')
        self.ax.set_title('Real-time Wavelet Coefficients - All 6 Bands', color='white')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # Style the axes
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
            
    def decode_packet(self, data):
        """Decode a binary packet and return all 6 coefficient values."""
        # Expect 26 bytes: header (1) + data (6x4) + footer (1)
        if len(data) != 26:
            return None
            
        header = data[0]
        footer = data[25]
        
        # Check header and footer
        if header != 0x07 or footer != 0x18:
            return None
            
        # Extract the 6x 32-bit signed integers (big-endian)
        try:
            values = []
            for i in range(6):
                start_idx = 1 + i * 4
                end_idx = start_idx + 4
                value_bytes = data[start_idx:end_idx]
                value = struct.unpack('>i', value_bytes)[0]
                values.append(value)
            return values
        except struct.error:
            return None
            
    def serial_reader(self):
        """Read data from serial port in a separate thread."""
        try:
            self.serial_conn = pyserial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} at {self.baudrate} baud")
            
            buffer = bytearray()
            packet_count = 0
            start_time = time.time()
            
            while self.running:
                if self.serial_conn.in_waiting > 0:
                    # Read available data
                    new_data = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer.extend(new_data)
                    
                    # Process complete packets
                    while len(buffer) >= 26:
                        # Look for header byte
                        header_idx = buffer.find(0x07)
                        if header_idx == -1:
                            # No header found, clear buffer
                            buffer.clear()
                            break
                            
                        # Remove data before header
                        if header_idx > 0:
                            buffer = buffer[header_idx:]
                            
                        # Check if we have a complete packet
                        if len(buffer) >= 26:
                            packet = buffer[:26]
                            values = self.decode_packet(packet)
                            
                            if values is not None:
                                current_time = time.time() - start_time
                                
                                with self.data_lock:
                                    self.times.append(current_time)
                                    self.coefficients['A11'].append(values[0])  # A11
                                    self.coefficients['D11'].append(values[1])  # D11
                                    self.coefficients['D10'].append(values[2])  # D10
                                    self.coefficients['D9'].append(values[3])   # D9
                                    self.coefficients['D8'].append(values[4])   # D8
                                    self.coefficients['D7'].append(values[5])   # D7
                                    
                                packet_count += 1
                                if packet_count % 100 == 0:
                                    print(f"Received {packet_count} packets, latest A11: {values[0]}, D11: {values[1]}")
                                    
                            # Remove processed packet
                            buffer = buffer[26:]
                        else:
                            break
                            
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
                
        except Exception as e:
            print(f"Serial communication error: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print("Serial connection closed")
                
    def animate(self, frame):
        """Animation function called by matplotlib."""
        with self.data_lock:
            if len(self.coefficients['A11']) > 0:
                # Update all lines
                x_data = list(range(len(self.coefficients['A11'])))
                
                self.line_a11.set_data(x_data, list(self.coefficients['A11']))
                self.line_d11.set_data(x_data, list(self.coefficients['D11']))
                self.line_d10.set_data(x_data, list(self.coefficients['D10']))
                self.line_d9.set_data(x_data, list(self.coefficients['D9']))
                self.line_d8.set_data(x_data, list(self.coefficients['D8']))
                self.line_d7.set_data(x_data, list(self.coefficients['D7']))
                
                # Auto-scale x-axis to show recent data
                if len(x_data) > self.max_points:
                    self.ax.set_xlim(len(x_data) - self.max_points, len(x_data))
                else:
                    self.ax.set_xlim(0, max(self.max_points, len(x_data)))
                    
        return [self.line_a11, self.line_d11, self.line_d10, self.line_d9, self.line_d8, self.line_d7]
        
    def start(self):
        """Start the plotter."""
        self.running = True
        
        # Start serial reader thread
        self.serial_thread = threading.Thread(target=self.serial_reader)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        # Start animation
        try:
            ani = animation.FuncAnimation(
                self.fig, self.animate, interval=50, blit=False, cache_frame_data=False
            )
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the plotter and cleanup."""
        self.running = False
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print("Plotter stopped")

def main():
    """Main function to parse arguments and start the plotter."""
    # Default values
    default_port = '/dev/cu.usbmodem178685801'
    default_baud = 115200
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = default_port
        
    if len(sys.argv) > 2:
        try:
            baudrate = int(sys.argv[2])
        except ValueError:
            print(f"Invalid baudrate: {sys.argv[2]}, using default {default_baud}")
            baudrate = default_baud
    else:
        baudrate = default_baud
    
    print(f"Starting Wavelet Coefficients Plotter (All 6 Bands)")
    print(f"Port: {port}")
    print(f"Baudrate: {baudrate}")
    print("Press Ctrl+C to stop")
    
    # Create and start plotter
    plotter = WaveletPlotter(port, baudrate)
    plotter.start()

if __name__ == "__main__":
    main()