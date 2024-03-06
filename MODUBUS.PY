# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import serial
import sys

# Setup serial connection
ser = serial.Serial(port='COM4', baudrate=9600, parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)

def send_command(command):
    """
    Send a command over serial.
    :param command: Command to send (as a list of hex values).
    :return: None
    """
    # Convert the command list to bytes
    command_bytes = bytes(command)
    # Write the command to the serial port
    ser.write(command_bytes)
    print(f"Sent: {' '.join(format(x, '02X') for x in command)}")

try:
    while True:
        # Wait for user input
        key = input("Press 'Y' to turn on the relay, 'N' to turn off, or 'E' to exit: ").upper()
        if key == 'Y':
            # Command to turn on the relay
            on_command = [0xFF, 0x0F, 0x00, 0x00, 0x00, 0x08, 0x01, 0xFF, 0x30, 0x1D]
            send_command(on_command)
        elif key == 'N':
            # Command to turn off the relay
            off_command = [0xFF, 0x0F, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00, 0x70, 0x5D]
            send_command(off_command)
        elif key == 'E':
            print("Exiting...")
            break
        else:
            print("Invalid input. Please press 'Y', 'N', or 'E'.")
finally:
    # Close the serial connection
    ser.close()
