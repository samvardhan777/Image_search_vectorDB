import os
import cv2
import uuid
import base64
import clear_all
import weaviate

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
if not WEAVIATE_URL:
    WEAVIATE_URL = 'http://localhost:8080'

print(WEAVIATE_URL, flush=True)

client = weaviate.Client(WEAVIATE_URL)


schema = {
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
    
client.schema.create_class(schema)
    
print("The schema has been defined.")