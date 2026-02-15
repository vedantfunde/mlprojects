import matplotlib.pyplot as plt

def plot(history):
    plt.plot(history.history['accuracy'],color='red',label='train')
    plt.plot(history.history['val_accuracy'],color='blue',label='test')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    print("Plot saved as accuracy_plot.png")