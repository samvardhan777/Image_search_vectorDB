import os
from create_schema import create_schema
import os
import cv2
import uuid
import base64
import ipyplot
import weaviate
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
import faiss
mtcnn = MTCNN()
dimension = 512 
index = faiss.IndexFlatL2(dimension)
model = InceptionResnetV1(pretrained='vggface2').eval()
TEST_DIR = "test"
image_embeddings_base64 = {}


DATA_DIR = "test_sample"
IMAGE_DIM = (100, 100)

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
if not WEAVIATE_URL:
    WEAVIATE_URL = 'http://localhost:8080'


client = weaviate.Client(WEAVIATE_URL)

create_schema(client)

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
        
def insert_data(client):
    for fn in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, fn)
        img = prepare_image(fp)
        embedding = create_embedding(img)
        if embedding is not None:
            b64_string = base64.b64encode(embedding[1]).decode('utf-8')
            data_properties = {
                "path": fn,
                "image": b64_string,
            }
            r = client.data_object.create(data_properties, "Stamp", str(uuid.uuid4()))
            print(fp, r)    

insert_data(client)    


def weaviate_img_search(img_str):

    sourceImage = {"image": img_str}

    weaviate_results = client.query.get(
       "Stamp", ["path", "_additional { certainty }"]
        ).with_near_image(
            sourceImage, encode=False
        ).with_limit(1).do()

    return weaviate_results["data"]["Get"]


for fn in os.listdir(TEST_DIR):
    fp = os.path.join(TEST_DIR, fn)  # Corrected path to use TEST_DIR
    img = prepare_image(fp)
    embedding = create_embedding(img)
    if embedding is not None:
        b64_string = base64.b64encode(embedding[1]).decode('utf-8')
print(weaviate_img_search(b64_string))

