import pyautogui
import time
import os

# Directory to save screenshots
save_directory = "screenshots"

# Create directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Interval between screenshots in seconds
interval = 1

try:
    screenshot_count = 0
    while True:
        # Capture the screenshot
        screenshot = pyautogui.screenshot()

        # Construct file name with timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"screenshot_{timestamp}.png"

        # Save screenshot to the specified directory
        screenshot.save(os.path.join(save_directory, file_name))

        print(f"Screenshot saved: {file_name}")
        screenshot_count += 1

        # Wait for the next interval
        time.sleep(interval)
except KeyboardInterrupt:
    print("Screenshot capturing stopped.")
