from src.dataset import load_data,normalize,clean_images
from src.model import build_model
from src.dataviz import plot

train_path = '/mnt/d/BTECH/ML/dogvcat/data/train'
test_path = '/mnt/d/BTECH/ML/dogvcat/data/test'
clean_images(train_path)
clean_images(test_path)
train_ds,test_ds = load_data(train_path,test_path)
train_ds,test_ds = normalize(train_ds,test_ds)
model = build_model()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=10,validation_data=test_ds)
print(history.history)
plot(history)