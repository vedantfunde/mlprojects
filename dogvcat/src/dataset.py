import keras
import tensorflow as tf

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

def process(image,label):
    image = tf.cast(image/255.0,tf.float32)
    return image,label


def normalize(train_ds,test_ds):
    train_ds = train_ds.map(process)
    test_ds = test_ds.map(process)
    return train_ds,test_ds
    
