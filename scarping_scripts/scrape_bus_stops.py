import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR


def visualize_ocr_results(img_path, ocr_result):
    # Load the image
    img = cv2.imread(img_path)
    
    # Extract bounding boxes and texts
    bboxes, texts = ocr_result[0], ocr_result[1]

    for bbox, (text, confidence) in zip(bboxes, texts):
        bbox = np.array(bbox, dtype=np.int32)  # Convert bbox to integer
        bbox = bbox.reshape((-1, 1, 2))  # Ensure proper shape for OpenCV
        
        # Draw bounding box
        cv2.polylines(img, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)

        # Get text position (above the bounding box)
        x, y = bbox[0][0]
        cv2.putText(img, f"{text} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def crop_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # Crop top and bottom
    img = img[450:, :] # app top half
    img = img[:-100, :] # phone's os bottom
    img = img[100:, :] # table header

    # splitting image into sr no, stop name, stop code
    sr_no_img = img[:, :211]
    stop_name_img = img[:, 211:-320]
    stop_code_img = img[:, -320:]

    return sr_no_img, stop_name_img, stop_code_img


def preprocess_img(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply dilation to merge close text regions
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    return img 


def process_with_ocr(img):
    ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr_model(img)
    return result


def merge_nearby_bboxes(ocr_result, threshold=30):
    bboxes, texts = ocr_result[0], ocr_result[1]

    merged_bboxes = []
    merged_texts = []

    # Sort by top-left Y coordinate
    sorted_indices = sorted(range(len(bboxes)), key=lambda i: np.min(bboxes[i][:, 1]))

    current_bbox = None
    current_text = ""

    for idx in sorted_indices:
        bbox = np.array(bboxes[idx], dtype=np.int32)
        text, confidence = texts[idx]

        if current_bbox is None:
            # Initialize with the first bbox
            current_bbox = bbox
            current_text = text
        else:
            # Compute vertical overlap or gap
            prev_y_max = np.max(current_bbox[:, 1])  # Bottom of previous bbox
            curr_y_min = np.min(bbox[:, 1])  # Top of current bbox

            if curr_y_min - prev_y_max <= threshold:  # Check vertical proximity
                # Merge bounding box (expand min/max coordinates)
                x_min = min(np.min(current_bbox[:, 0]), np.min(bbox[:, 0]))
                y_min = min(np.min(current_bbox[:, 1]), np.min(bbox[:, 1]))
                x_max = max(np.max(current_bbox[:, 0]), np.max(bbox[:, 0]))
                y_max = max(np.max(current_bbox[:, 1]), np.max(bbox[:, 1]))
                
                current_bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                
                # Concatenate text
                current_text += " " + text
            else:
                # Store previous bbox & text and start new one
                merged_bboxes.append(current_bbox)
                merged_texts.append((current_text, confidence))

                current_bbox = bbox
                current_text = text

    # Append last merged bbox
    if current_bbox is not None:
        merged_bboxes.append(current_bbox)
        merged_texts.append((current_text, confidence))

    return merged_bboxes, merged_texts

def main():
    result = {
        0: [],
        1: [],
        2: []
    }

    input_folder = "bus_stop_input_data/"
    
    for img_name in sorted(os.listdir(input_folder))[81:]:
        ext = os.path.splitext(img_name)[-1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            continue

        if '81' not in img_name:
            continue

        print(img_name)

        print(f"PROCESSING: image {img_name}")
        cropped_images = crop_image(input_folder + img_name)
        if cropped_images is None:
            print(f"ERROR: image {img_name} None after cropped")
            continue

        fig, ax = plt.subplots(1, 3)

        for i in range(len(cropped_images)):
            plt.subplot(1, 3, i+1)
            plt.imshow(cropped_images[i])
        

        sr_no = None
        stop_names = None
        stop_codes = None

        for i in range(len(cropped_images)):
            processed_img = preprocess_img(cropped_images[i])
            ocr_result = process_with_ocr(processed_img)

            merged_ocr_result = merge_nearby_bboxes(ocr_result, 30)

            bboxes, texts = merged_ocr_result[0], merged_ocr_result[1]

            for bbox, (text, confidence) in zip(bboxes, texts):
                bbox = np.array(bbox, dtype=np.int32)  # Convert bbox to integer
                bbox = bbox.reshape((-1, 1, 2))  # Ensure proper shape for OpenCV
                
                # Draw bounding box
                cv2.polylines(cropped_images[i], [bbox], isClosed=True, color=(0, 255, 0), thickness=2)

                # Get text position (above the bounding box)
                x, y = bbox[0][0]
                cv2.putText(cropped_images[i], f"{text} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            plt.subplot(1, 3, i+1)
            plt.imshow(cropped_images[i])

            if i == 0:
                sr_no = [text for text, confidence in merged_ocr_result[1]]
            elif i == 1:
                stop_names = [text for text, confidence in merged_ocr_result[1]]
            elif i == 2:
                stop_codes = [text for text, confidence in merged_ocr_result[1]]

        # if len(sr_no) != len(stop_names) or len(sr_no) != len(stop_codes):
        #     print(f"ERROR: image {img_name} - ARRAY LENGTHS NOT EQUAL")
        # else:
        result[0].extend(sr_no)
        result[1].extend(stop_names)  # Extend result[1] with the updated list
        result[2].extend(stop_codes)

        plt.title(img_name)
        plt.show()



    with open("bus_stop_data_error.pkl", "wb") as f:
        pickle.dump(result, f)

    print(result)


    df = pd.DataFrame(result)
    df.to_csv("bus_stop_data_error.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()

    

