#!/usr/bin/env python3
"""
Serial Data Broker with Virtual Ports

This broker creates two virtual serial ports and forwards all data from the real
serial port to both virtual ports in real-time. This allows multiple applications
to simultaneously access the same serial data stream without conflicts.

Usage:
    python3 serial_broker.py [real_port] [baudrate]
    
Example:
    python3 serial_broker.py /dev/cu.usbmodem178685801 115200
    
The broker will create two virtual ports (PTY pairs) and display their paths.
Your existing scripts can then connect to these virtual ports instead of the real port.
"""

import sys
import os
import pty
import select
import serial
import threading
import time
import signal
from typing import List, Optional

class SerialBroker:
    def __init__(self, real_port: str, baudrate: int = 115200):
        self.real_port = real_port
        self.baudrate = baudrate
        self.running = False
        
        # Real serial connection
        self.serial_conn: Optional[serial.Serial] = None
        
        # Virtual port pairs (master, slave)
        self.virtual_ports: List[tuple] = []
        self.virtual_port_paths: List[str] = []
        self.virtual_masters: List[int] = []
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self.forward_threads: List[threading.Thread] = []
        
        # Statistics
        self.bytes_received = 0
        self.bytes_forwarded = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.start_time = time.time()
        
    def create_virtual_ports(self, count: int = 2) -> List[str]:
        """Create virtual serial port pairs using PTY."""
        print(f"Creating {count} virtual serial ports...")
        
        for i in range(count):
            try:
                # Create PTY pair
                master, slave = pty.openpty()
                
                # Get the slave device path
                slave_path = os.ttyname(slave)
                
                self.virtual_ports.append((master, slave))
                self.virtual_port_paths.append(slave_path)
                self.virtual_masters.append(master)
                
                print(f"Virtual port {i+1}: {slave_path}")
                
                # Set master to non-blocking mode once at startup
                import fcntl
                flags = fcntl.fcntl(master, fcntl.F_GETFL)
                fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                # Close slave fd (clients will open it)
                os.close(slave)
                
            except Exception as e:
                print(f"Error creating virtual port {i+1}: {e}")
                self.cleanup()
                return []
        
        return self.virtual_port_paths
    
    def connect_real_serial(self) -> bool:
        """Connect to the real serial port."""
        try:
            print(f"Connecting to real serial port: {self.real_port}")
            self.serial_conn = serial.Serial(
                port=self.real_port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            print(f"Connected to {self.real_port} at {self.baudrate} baud")
            return True
            
        except Exception as e:
            print(f"Error connecting to real serial port: {e}")
            return False
    
    def forward_data_to_virtual_port(self, master_fd: int, port_index: int):
        """Forward data to a specific virtual port in a separate thread."""
        port_name = f"Virtual Port {port_index + 1}"
        bytes_written = 0
        
        while self.running:
            try:
                # Check if there are clients connected by trying to write
                # This is a simple way to detect if someone is reading from the slave
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.0001)  # 0.1ms delay
                
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    print(f"Error in {port_name} forwarding thread: {e}")
                break
        
        print(f"{port_name} forwarding thread stopped")
    
    def serial_reader_thread(self):
        """Main thread that reads from real serial and forwards to virtual ports."""
        print("Starting serial reader thread...")
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        error_counts = [0] * len(self.virtual_masters)  # Track errors per port
        
        while self.running:
            if not self.serial_conn or not self.serial_conn.is_open:
                if reconnect_attempts < max_reconnect_attempts:
                    print(f"Attempting to reconnect... (attempt {reconnect_attempts + 1})")
                    if self.connect_real_serial():
                        reconnect_attempts = 0
                    else:
                        reconnect_attempts += 1
                        time.sleep(2)  # Wait before retry
                        continue
                else:
                    print("Max reconnection attempts reached. Stopping.")
                    break
            
            try:
                # Read data from real serial port with larger buffer
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    if data:
                        self.bytes_received += len(data)
                        # Estimate packet count (assuming 26 bytes per packet)
                        self.packets_received += len(data) // 26
                        
                        # Forward data to all virtual ports simultaneously
                        ports_successful = 0
                        for i, master_fd in enumerate(self.virtual_masters):
                            try:
                                bytes_written = os.write(master_fd, data)
                                self.bytes_forwarded += bytes_written
                                error_counts[i] = 0  # Reset error count on success
                                ports_successful += 1
                                    
                            except (OSError, IOError) as e:
                                error_counts[i] += 1
                                # Track dropped packets
                                if error_counts[i] == 1:
                                    print(f"Virtual port {i+1} ({self.virtual_port_paths[i]}): No client connected")
                                elif error_counts[i] % 1000 == 0:  # Print every 1000 errors
                                    print(f"Virtual port {i+1}: Still no client ({error_counts[i]} attempts)")
                        
                        # Track packets that couldn't be forwarded to any port
                        if ports_successful == 0:
                            self.packets_dropped += len(data) // 26
                
                # No sleep - run as fast as possible to avoid dropping data
                
            except Exception as e:
                if self.running:
                    print(f"Error in serial reader: {e}")
                    # Try to reconnect
                    if self.serial_conn:
                        self.serial_conn.close()
                    self.serial_conn = None
                    time.sleep(1)
        
        print("Serial reader thread stopped")
    
    def print_statistics(self):
        """Print broker statistics."""
        while self.running:
            time.sleep(5)  # Print stats every 5 seconds
            
            if self.bytes_received > 0:
                elapsed = time.time() - self.start_time
                rate = self.bytes_received / elapsed if elapsed > 0 else 0
                packet_rate = self.packets_received / elapsed if elapsed > 0 else 0
                drop_rate = (self.packets_dropped / self.packets_received * 100) if self.packets_received > 0 else 0
                
                print(f"[STATS] Running for {elapsed:.1f}s | "
                      f"Packets: {self.packets_received:,} received, {self.packets_dropped:,} dropped ({drop_rate:.1f}%) | "
                      f"Rate: {packet_rate:.1f} packets/sec | "
                      f"Bytes: {self.bytes_received:,} received, {self.bytes_forwarded:,} forwarded")
    
    def start(self):
        """Start the broker."""
        print("Serial Broker Starting...")
        print("=" * 50)
        
        # Create virtual ports
        virtual_paths = self.create_virtual_ports(2)
        if not virtual_paths:
            print("Failed to create virtual ports")
            return False
        
        # Connect to real serial port
        if not self.connect_real_serial():
            print("Failed to connect to real serial port")
            self.cleanup()
            return False
        
        print("\n" + "=" * 60)
        print("BROKER READY!")
        print("=" * 60)
        print("Virtual ports created:")
        for i, path in enumerate(virtual_paths):
            print(f"  Port {i+1}: {path}")
        
        print(f"\nNOTE: You'll see 'No client connected' messages until you")
        print(f"      connect your scripts to the virtual ports.")
        
        print("\nTo use the virtual ports, run these commands in separate terminals:")
        print("\n📊 SERIAL PLOTTER:")
        print(f"   python3 serial_plotter.py {virtual_paths[0]}")
        print("\n📹 LIVE VISUALIZER:")
        print(f"   python3 live_visualizer.py {virtual_paths[1]}")
        
        print(f"\n🔄 The broker will forward all data from the real port to both virtual ports.")
        print(f"📡 Both scripts will receive identical real-time data streams.")
        print(f"\n❌ Press Ctrl+C to stop the broker")
        print("=" * 60)
        
        # Start broker
        self.running = True
        
        # Start serial reader thread
        self.read_thread = threading.Thread(target=self.serial_reader_thread, daemon=True)
        self.read_thread.start()
        
        # Start statistics thread
        stats_thread = threading.Thread(target=self.print_statistics, daemon=True)
        stats_thread.start()
        
        return True
    
    def stop(self):
        """Stop the broker and cleanup."""
        print("\nStopping broker...")
        self.running = False
        
        # Wait for threads to finish
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        
        for thread in self.forward_threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        self.cleanup()
        print("Broker stopped")
    
    def cleanup(self):
        """Clean up resources."""
        # Close real serial connection
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Real serial connection closed")
        
        # Close virtual port masters
        for master_fd in self.virtual_masters:
            try:
                os.close(master_fd)
            except:
                pass
        
        print("Virtual ports closed")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nReceived interrupt signal...")
    if 'broker' in globals():
        broker.stop()
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Default values
    default_port = '/dev/cu.usbmodem178685801'
    default_baud = 115200
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        real_port = sys.argv[1]
    else:
        real_port = default_port
        
    if len(sys.argv) > 2:
        try:
            baudrate = int(sys.argv[2])
        except ValueError:
            print(f"Invalid baudrate: {sys.argv[2]}, using default {default_baud}")
            baudrate = default_baud
    else:
        baudrate = default_baud
    
    print("Serial Data Broker with Virtual Ports")
    print("=" * 50)
    print(f"Real port: {real_port}")
    print(f"Baudrate: {baudrate}")
    print("Creating 2 virtual ports for data forwarding...")
    
    # Create and start broker
    global broker
    broker = SerialBroker(real_port, baudrate)
    
    if broker.start():
        try:
            # Keep main thread alive
            while broker.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            broker.stop()
    else:
        print("Failed to start broker")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
