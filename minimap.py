import cv2
import csv
import numpy as np

# Load your minimap image
minimap = cv2.imread('pitch.png')

def apply_homography(src, H):
  '''Applies a homography H to the source points.
   Input:
      src: source points, shape (N, 2)
      H: homography from source points to destination points, shape (3, 3)
   Output:
     dst: destination points, shape (N, 2)
  '''
  N = src.shape[0]

  src = np.append(src,np.ones([N,1]),1)

  dst = H @ src.T

  dst = dst/dst[2,:]

  return dst.T[:,:2]

# Define colors for different classes
colors = {
    'player': (255, 255, 255),  # White
    'goalkeeper': (255, 0, 0),  # Red
    'referee': (0, 0, 255),  # Blue
    'ball': (255, 255, 0)  # Yellow
    # Add more classes and colors as needed
}

# Define points on the actual field in the camera view and corresponding points on the minimap
# These points should be in the format [(x1, y1), (x2, y2), ...]
#field_points_camera_view = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
#field_points_minimap = np.array([[x1', y1'], [x2', y2'], [x3', y3'], [x4', y4']], dtype='float32')

# Compute the homography matrix
#H, _ = cv2.findHomography(field_points_camera_view, field_points_minimap)

center_points = []
class_names = []

# Read the CSV file
with open('results\scene00251.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        # Extract center_x and center_y from each row
        center_x, center_y = int(row[5]), int(row[6])  # Assuming these are in the 6th and 7th columns
        class_name = row[0] 
        center_points.append([center_x, center_y])
        class_names.append(class_name)

# Convert the list to a NumPy array
center_points = np.array(center_points, dtype='float32')

H = np.array([[-1.67696508e+00, -1.51902798e+00,  1.12661142e+03],
              [-5.43165070e-01, -1.33366972e+00,  6.95421345e+02],
              [-1.16671212e-03, -1.86942256e-03,  1.00000000e+00]])

H_inv = np.linalg.inv(H)

# Apply the homography to transform the points
transformed_points = apply_homography(center_points, H_inv)
#transformed_points = cv2.perspectiveTransform(np.array([center_points]), H)[0]

# Draw the points and class names on the minimap
for point, class_name in zip(transformed_points, class_names):
    x, y = point.ravel()
    color = colors.get(class_name, (255, 255, 255))  # Default to white if class not found
    cv2.circle(minimap, (int(x), int(y)), 5, color, -1)
    cv2.putText(minimap, class_name, (int(x), int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Save or display the minimap
cv2.imwrite('minimap_with_players.jpg', minimap)
# or cv2.imshow('Minimap', minimap) and cv2.waitKey(0)
