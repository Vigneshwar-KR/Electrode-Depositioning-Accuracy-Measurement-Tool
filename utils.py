import cv2
import numpy as np
import math
import pandas as pd
import os
from tkinter import Tk, filedialog
from sklearn.cluster import KMeans

def resize_image(image, screen_width=1366, screen_height=768):
    h, w = image.shape[:2]
    scaling_factor = min(screen_width / float(w), screen_height / float(h))
    resized_image = cv2.resize(image, (int(w * scaling_factor), int(h * scaling_factor)), interpolation=cv2.INTER_AREA)
    return resized_image

def find_midpoint(group):
    mid_x = int(np.mean([circle[0] for circle in group]))
    mid_y = int(np.mean([circle[1] for circle in group]))
    return (mid_x, mid_y)

def process_images(folder_path):
    all_data = []
    # Create the 'detected_circles' folder
    output_folder = os.path.join(folder_path, 'detected_circles')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        if image_name.lower().endswith(('.jpg', '.jpeg')):
            image = cv2.imread(image_path)
            image = resize_image(image)  

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            dp = 10/10
            minDist = 20
            param1 = 50
            param2 = 30
            minRadius = 20
            maxRadius = 30

            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

            mid_group1_x, mid_group1_y = None, None
            mid_group2_x, mid_group2_y = None, None
            mid_group3_x, mid_group3_y = None, None
            distance_1_2_x, distance_1_2_y = None, None
            distance_1_3_x, distance_1_3_y = None, None
            distance_1_2_x_in_mm, distance_1_2_y_in_mm = None, None
            distance_1_3_x_in_mm, distance_1_3_y_in_mm = None, None

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                if len(circles) >= 12:
                    kmeans = KMeans(n_clusters=3)
                    kmeans.fit(circles[:, :2])  
                    labels = kmeans.labels_

                    group1 = circles[labels == 0]
                    group2 = circles[labels == 1]
                    group3 = circles[labels == 2]

                    mid_group1 = find_midpoint(group1)
                    mid_group2 = find_midpoint(group2)
                    mid_group3 = find_midpoint(group3)

                    groups = sorted([(mid_group1, group1), (mid_group2, group2), (mid_group3, group3)], key=lambda x: x[0][0])

                    mid_group1, group1 = groups[0]
                    mid_group2, group2 = groups[1]
                    mid_group3, group3 = groups[2]
                    
                    mid_group1_x, mid_group1_y = mid_group1
                    mid_group2_x, mid_group2_y = mid_group2
                    mid_group3_x, mid_group3_y = mid_group3

                    for (x, y, r) in circles:
                        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red dot - at the center

                    cv2.circle(image, mid_group1, 5, (255, 0, 0), -1)  # Blue dot - midpoint of group 1
                    cv2.circle(image, mid_group2, 5, (0, 255, 0), -1)  # Green dot - midpoint of group 2
                    cv2.circle(image, mid_group3, 5, (0, 0, 255), -1)  # Red dot - midpoint of group 3

                    # For values with directional info use this
                    distance_1_2_x = mid_group2_x - mid_group1_x
                    distance_1_2_y = mid_group2_y - mid_group1_y
                    distance_1_3_x = mid_group3_x - mid_group1_x
                    distance_1_3_y = mid_group3_y - mid_group1_y
                    # # For absolute values use this
                    # distance_1_2_x = abs(mid_group2_x - mid_group1_x)
                    # distance_1_2_y = abs(mid_group2_y - mid_group1_y)
                    # distance_1_3_x = abs(mid_group3_x - mid_group1_x)
                    # distance_1_3_y = abs(mid_group3_y - mid_group1_y)

                    
                    conversion_factor = 0.07281  # our calculated value
                    distance_1_2_x_in_mm = distance_1_2_x * conversion_factor
                    distance_1_2_y_in_mm = distance_1_2_y * conversion_factor
                    distance_1_3_x_in_mm = distance_1_3_x * conversion_factor
                    distance_1_3_y_in_mm = distance_1_3_y * conversion_factor

                else:
                    print(f"Unexpected number of circles in image {image_name}: {len(circles)}")
                    continue

            all_data.append({
                'Image File': image_name,
                'mid_group1_x (Base) [pixels]': mid_group1_x,
                'mid_group1_y (Base) [pixels]': mid_group1_y,
                'mid_group2_x [pixels]': mid_group2_x,
                'mid_group2_y [pixels]': mid_group2_y,
                'mid_group3_x [pixels]': mid_group3_x,
                'mid_group3_y [pixels]': mid_group3_y,
                'Distance Group 1-2 X [pixels]': distance_1_2_x,
                'Distance Group 1-2 Y [pixels]': distance_1_2_y,
                'Distance Group 1-3 X [pixels]': distance_1_3_x,
                'Distance Group 1-3 Y [pixels]': distance_1_3_y,
                'Distance Group 1-2 X [mm]': distance_1_2_x_in_mm,
                'Distance Group 1-2 Y [mm]': distance_1_2_y_in_mm,
                'Distance Group 1-3 X [mm]': distance_1_3_x_in_mm,
                'Distance Group 1-3 Y [mm]': distance_1_3_y_in_mm
            })

            # resized_image = resize_image(image, screen_width=800, screen_height=600)
            # cv2.imshow(f'Hough Circle Detection - {image_name}', resized_image)

            # cv2.waitKey(0)

            # cv2.destroyAllWindows()
            # Save the processed image to the 'detected_circles' folder

            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, image)


    df = pd.DataFrame(all_data)
    output_path = os.path.join(folder_path, 'circle_detection_results.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Data exported to {output_path}")