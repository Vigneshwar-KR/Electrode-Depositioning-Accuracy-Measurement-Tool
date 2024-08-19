import cv2
import numpy as np
import math

# Function to resize the image to fit within the screen
def resize_image_to_fit_screen(image, screen_width=1366, screen_height=768):
    h, w = image.shape[:2]

    # Calculate the scaling factor to fit the image within the screen
    scaling_factor = min(screen_width / float(w), screen_height / float(h))

    # Resize the image
    resized_image = cv2.resize(image, (int(w * scaling_factor), int(h * scaling_factor)), interpolation=cv2.INTER_AREA)
    
    return resized_image
import cv2
import numpy as np

# Load the image
image_path = 'C:/Vicky/Tu Braunschweig/HIWI-IWF/Electrode plate task/Bilder_Positionierung/Calib/IMG_20240816_105801.jpg'
image = cv2.imread(image_path)
image = resize_image_to_fit_screen(image)  # Resize the image to fit within the screen


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Parameters for Hough Circle Transform
dp = 1.0
minDist = 20
param1 = 50
param2 = 30
minRadius = 30
maxRadius = 50

# Apply Hough Circle Transform to detect circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                           param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

# Clone the original image to draw circles
output = image.copy()

# Ensure at least some circles were found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    # Sort circles based on the y-coordinate (from top to bottom)
    circles_sorted_by_y = sorted(circles, key=lambda x: x[1])
    
    # Check if there are at least two circles detected
    if len(circles_sorted_by_y) >= 2:
        # Select the last two circles based on the y-coordinate
        selected_circles = [circles_sorted_by_y[-1], circles_sorted_by_y[-2]]  # Last two circles

        # Calculate the Euclidean distance between the two circles
        x1, y1 = selected_circles[0][:2]
        x2, y2 = selected_circles[1][:2]
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        print(f"Distance between the two circles: {pixel_distance:.2f} pixels")
        
        # Calculate the conversion factor from pixels to millimeters
        real_distance_mm = 15  # Real-world distance between the two circles in millimeters
        pixel_to_mm_factor = real_distance_mm / pixel_distance
        print(f"Conversion factor: {pixel_to_mm_factor:.5f} mm/pixel")

        # Print the details of the selected circles
        for (x, y, r) in selected_circles:
            print(f"Detected Circle - X: {x}, Y: {y}, Radius: {r}")
            
            # Draw the outer circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)  # Red dot at the center

    else:
        print("Not enough circles detected to select the last two circles.")
else:
    print("No circles detected.")

# Resize the image to fit the window
resized_image = resize_image_to_fit_screen(output, screen_width=800, screen_height=600)

# Create a window to display the image
cv2.namedWindow('Hough Circle Detection', cv2.WINDOW_NORMAL)

# Display the image with circles
cv2.imshow('Hough Circle Detection', resized_image)

# Wait until a key is pressed to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
