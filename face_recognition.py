import cv2 as cv
import numpy as np
import os

def visualize(image, faces, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
            for i in range(5):
                cv.circle(image, (coords[4 + 2*i], coords[5 + 2*i]), 2, (0, 255, 0), thickness)

def main():
    reference_image_path = "aadhaar_images/reference.jpg"
    query_images_dir = "sample_images/"

    print(f"Loading reference image from: {reference_image_path}")
    reference_image = cv.imread(reference_image_path)
    
    if reference_image is None:
        print(f"Error: Could not load reference image from {reference_image_path}")
        return

    print("Reference image loaded successfully.")

    score_threshold = 0.9
    nms_threshold = 0.3
    top_k = 5000
    face_detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "",
                                             (reference_image.shape[1], reference_image.shape[0]), 
                                             score_threshold, nms_threshold, top_k)
    recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

    face_in_aadhaar = face_detector.detect(reference_image)
    if face_in_aadhaar[1] is None:
        print("No face detected in the reference Aadhaar image.")
        return

    print(f"Detected {len(face_in_aadhaar[1])} faces in the reference image.")

    aadhaar_face_align = recognizer.alignCrop(reference_image, face_in_aadhaar[1][0])
    aadhaar_face_feature = recognizer.feature(aadhaar_face_align)

    visualize(reference_image, face_in_aadhaar)
    
    cv.imshow("Reference Image", reference_image)
    cv.waitKey(0) 

    cosine_similarity_threshold = 0.363
    l2_similarity_threshold = 1.128

    correct_detections = 0
    correct_matches = 0
    total_images = 0

    for image_name in os.listdir(query_images_dir):
        query_image_path = os.path.join(query_images_dir, image_name)
        print(f"Processing query image: {query_image_path}")
        query_image = cv.imread(query_image_path)
        
        if query_image is None:
            print(f"Error: Could not load query image from {query_image_path}")
            continue

        face_detector.setInputSize((query_image.shape[1], query_image.shape[0]))
        face_in_query = face_detector.detect(query_image)

        if face_in_query[1] is not None:
            query_face_align = recognizer.alignCrop(query_image, face_in_query[1][0])
            query_face_feature = recognizer.feature(query_face_align)

            cosine_score = recognizer.match(aadhaar_face_feature, query_face_feature, cv.FaceRecognizerSF_FR_COSINE)
            l2_score = recognizer.match(aadhaar_face_feature, query_face_feature, cv.FaceRecognizerSF_FR_NORM_L2)

            if cosine_score >= cosine_similarity_threshold or l2_score <= l2_similarity_threshold:
                correct_matches += 1

            correct_detections += 1

            visualize(query_image, face_in_query)

            cv.imshow("Query Image", query_image)
            cv.waitKey(0) 

        total_images += 1

    detection_accuracy = (correct_detections / total_images) * 100 if total_images > 0 else 0
    matching_accuracy = (correct_matches / total_images) * 100 if total_images > 0 else 0

    print(f"Detection Accuracy: {detection_accuracy:.2f}%")
    print(f"Matching Accuracy: {matching_accuracy:.2f}%")
    print(f"Number of matches: {correct_matches} out of {total_images} images")

if __name__ == "__main__":
    main()
