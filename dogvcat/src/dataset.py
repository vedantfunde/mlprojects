import keras
import tensorflow as tf
import os

def clean_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img_bytes = tf.io.read_file(path)
                    tf.io.decode_image(img_bytes)
                except Exception:
                    print(f"Removing corrupt image: {path}")
                    os.remove(path)
                    count += 1
    print(f"Cleaned {count} corrupt images.")


def load_data(train_path,test_path):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=train_path,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256,256)
    )

    test_ds = keras.utils.image_dataset_from_directory(
        directory=test_path,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256,256)
    )
    return train_ds,test_ds

def normalize(train_ds, test_ds):
    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
    return train_ds, test_ds
    
