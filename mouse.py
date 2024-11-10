from pynput import mouse
import time
import threading
import csv

# Variables to store mouse position and click status
mouse_position = (0, 0)
mouse_clicked = False
mouse_click_type = None

# Sampling rate in Hz
sampling_rate = 250
sampling_interval = 1 / sampling_rate  # Time interval between samples

# File path to store mouse events
file_path = "mouse_events.csv"

# Function to capture mouse movement
def on_move(x, y):
    global mouse_position
    mouse_position = (x, y)

# Function to capture mouse clicks
def on_click(x, y, button, pressed):
    global mouse_clicked, mouse_click_type
    mouse_clicked = pressed
    mouse_click_type = button if pressed else None

# Listener thread for mouse events
def start_mouse_listener():
    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        listener.join()

# Start mouse listener in a separate thread
listener_thread = threading.Thread(target=start_mouse_listener)
listener_thread.start()

# Open file to log the events
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Timestamp", "Position_X", "Position_Y", "Clicked", "Button"])
    
    try:
        while True:
            # Record the current time
            start_time = time.time()
            
            # Capture the current mouse data
            current_position = mouse_position
            current_clicked = mouse_clicked
            current_click_type = mouse_click_type

            # Write data to the file
            writer.writerow([f"{start_time:.3f}", current_position[0], current_position[1], current_clicked, current_click_type])
            
            # Wait until the next sample time
            time.sleep(sampling_interval)
    except KeyboardInterrupt:
        print("Mouse activity logging stopped.")

# Join listener thread
listener_thread.join()
