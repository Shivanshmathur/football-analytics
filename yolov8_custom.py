import cv2
import os
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

"""
# For video
# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
"""
# Define path to directory containing images and videos for inference
source = 'test\images'
test_img = 'scene00251.png'
# Run batched inference on a list of images
results = model(test_img)  # return a list of Results objects


# Process results list
for result in results:
    filename, extension = os.path.splitext(os.path.basename(result.path))
    txt_file = os.path.join('results', filename) + '.txt'
    csv_file = os.path.join('results', filename) + '.csv'
    jpg_file = os.path.join('results', filename) + '.jpg'
    #print(result.names)
    result.save_txt(txt_file)

    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    im.save(jpg_file)  # save image
    
    with open(csv_file, 'w') as file:
        file.write("class_name,x_min,y_min,x_max,y_max,center_x,center_y\n")

        boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs
        for box in boxes:                                          # iterate boxes
            x_min, y_min, x_max, y_max = box.xyxy[0].astype(int)                       # get corner points as int
            # Calculate center coordinates
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            # Get class name
            class_name = result.names[int(box.cls[0])]
            # Write formatted data to CSV file
            file.write(f"{class_name},{x_min},{y_min},{x_max},{y_max},{center_x},{center_y}\n")


    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

"""
# Open a file to write the results
with open('detections.txt', 'w') as file:
    for frame in results:
        # Check if boxes are available
        if frame.boxes is not None:
            for box in frame.boxes:
                # Extract coordinates
                x_min, y_min, x_max, y_max = map(int, box.xyxy.tolist())

                # Get class ID and extract class name
                class_id = int(box.class_id)
                class_name = model.names[class_id]

                # Write formatted data to file
                file.write(f"{class_name}, {x_min}, {y_min}, {x_max}, {y_max}\n")
"""