#!/bin/bash
"""
Startup script for Serial Broker

This script starts the serial broker and provides instructions for using
the virtual ports with your existing scripts.
"""

# Configuration
REAL_PORT="/dev/cu.usbmodem178685801"
BAUDRATE="115200"
PYTHON_CMD="python3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Serial Broker Startup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python 3 is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if serial broker exists
if [ ! -f "serial_broker.py" ]; then
    echo -e "${RED}Error: serial_broker.py not found in current directory${NC}"
    exit 1
fi

# Check if the real serial port exists
if [ ! -e "$REAL_PORT" ]; then
    echo -e "${YELLOW}Warning: Real serial port $REAL_PORT not found${NC}"
    echo -e "${YELLOW}The broker will attempt to connect when the device becomes available${NC}"
    echo ""
fi

echo -e "${GREEN}Starting Serial Broker...${NC}"
echo -e "Real port: ${YELLOW}$REAL_PORT${NC}"
echo -e "Baudrate: ${YELLOW}$BAUDRATE${NC}"
echo ""

# Start the broker
echo -e "${BLUE}Running: $PYTHON_CMD serial_broker.py $REAL_PORT $BAUDRATE${NC}"
echo ""

# Execute the broker
exec $PYTHON_CMD serial_broker.py "$REAL_PORT" "$BAUDRATE"
