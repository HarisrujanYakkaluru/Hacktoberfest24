import cv2
import numpy as np 
import os

# Part 3 - Inference

def similarity_search(images_array, query_image):
    best_match_index = -1
    best_match_score = 0 

    for i in range(len(images_array)):
        norm2 = np.linalg.norm(images_array[i], axis=1)
        cosine_similarity = query_image@(images_array[i].T)
        score = 1 - cosine_similarity[0, 0]/(norm2)
        print(score[0])
        if score >= best_match_score:
            best_match_score = score
            best_match_index = i        

    return (best_match_index, best_match_score)


known_faces_array = np.load('known_faces.npy', allow_pickle=True)

sorted_list = sorted(os.listdir('dataset/'))

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) 
cap.set(3, 1080)
cap.set(4, 640)

while(True): 
    ret, image = cap.read() 
    cv2.namedWindow('Best Match')
    faces = face_detector.detectMultiScale(image, 1.2, 5)
    
    if len(faces) == 0:
        cv2.putText(image, 'NO MATCH FOUND', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.destroyWindow('Best Match')
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cropped_image = gray[y : y+h, x : x+w]
            query_image = cv2.resize(cropped_image, (100, 100)).reshape(1, -1)

            best_match_index, best_match_score = similarity_search(known_faces_array, query_image)
            name = f"{sorted_list[best_match_index].split('_')[0]}"
            cv2.rectangle(image, (x, y-35), (x+w, y), (0, 0, 255), cv2.FILLED)
            cv2.putText(image, name, (x+20, y-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.2, (255, 255, 255), 2)
            cv2.imshow('Best Match', cv2.imread(os.path.join('dataset', sorted_list[best_match_index])))

    cv2.imshow('Image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows()
