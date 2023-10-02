import os
import cv2
import uuid
import base64
import clear_all

DATA_DIR = "test_sample"
IMAGE_DIM = (100, 100)



def create_schema(client):
    class_obj = {
        "class": "Stamp",
        "description": "",
        "moduleConfig": {
            "img2vec-neural": {
                "imageFields": [
                    "image"
                ]
            }
        },
        "properties": [
            {
                "dataType": [
                    "blob"
                ],
                "description": "Image",
                "name": "image"
            },
            {
                "dataType": [
                    "string"
                ],
                "description": "",
                "name": "path"
            }
            
        ],
        "vectorIndexType": "hnsw",
        "vectorizer": "img2vec-neural"
    }
    
    client.schema.create_class(class_obj)
    
    print("The schema has been defined.")