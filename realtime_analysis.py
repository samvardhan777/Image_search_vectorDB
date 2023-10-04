import os
import cv2
import uuid
import base64
import ipyplot
import weaviate
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
import faiss
mtcnn = MTCNN()
dimension = 512 
index = faiss.IndexFlatL2(dimension)
model = InceptionResnetV1(pretrained='vggface2').eval()
TEST_DIR = "test"
IMAGE_DIM = (100, 100)
client = weaviate.Client("http://localhost:8080")
frame_skip_factor = 2 
cap = cv2.VideoCapture(0)
frame_counter = 0

schema = client.schema.get()
print(schema)

def prepare_image(file_path: str):
    """Read image from file_path
    and resize it to a fixed size square
    """
    img = cv2.imread(file_path)
    resized = cv2.resize(img, IMAGE_DIM, interpolation= cv2.INTER_LINEAR)
    return resized


def create_embedding(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = [int(i) for i in box]
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            jpg_img = cv2.imencode('.jpg', face)
            return jpg_img

def weaviate_img_search(img_str):

    sourceImage = {"image": img_str}

    weaviate_results = client.query.get(
       "Stamp", ["path", "_additional { certainty }"]
        ).with_near_image(
            sourceImage, encode=False
        ).with_limit(1).do()

    return weaviate_results["data"]["Get"]

while True:
    ret, frame = cap.read()
    frame_counter += 1

    if frame_counter % frame_skip_factor == 0:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x, y, w, h = [int(i) for i in box]
                face = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2) 
                embedding = create_embedding(face)
                if embedding is not None:
                    b64_string = base64.b64encode(embedding[1]).decode('utf-8')
                    result = weaviate_img_search(b64_string)
                    path=result['Stamp'][0]['path']
                    file_path = os.path.join('test_sample', path)
                    if os.path.exists(file_path):
                        print("The file exists.")
                        max_image = cv2.imread(file_path)
                        max_image = cv2.resize(max_image, (200, 200))
                        cv2.putText(frame, f'Celebrity: {path}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        max_image = cv2.resize(max_image, (200, 200))
                        frame[0:200, frame.shape[1]-200:frame.shape[1]] = max_image
                    else:
                        print("The file does not exist.")
                        max_image = np.zeros((200, 200, 3), dtype=np.uint8)
                        cv2.putText(max_image, 'No Image Found', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        frame[0:200, 0:200] = max_image

        cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break





