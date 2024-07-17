import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def recognize(index):
    switcher = {
        1: "博",
        2: "学",
        3: "笃",
        4: "志",
        5: "切",
        6: "问",
        7: "近",
        8: "思",
        9: "自",
        10: "由",
        11: "无",
        12: "用"
    }
    return switcher.get(index, "nothing")


def image_to_vector(i, j):
    image = Image.open("train_data/train/{}/{}.bmp".format(i, j))
    width, height = image.size
    gray_array = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gray = image.getpixel((x, y))
            gray_array[y, x] = 1 if gray == 255 else 0
    return gray_array

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class BPnet:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 256
        self.output_size = 12
        self.learning_rate = 0.01
        self.losses = []
        self.test_losses = []
        self.test_accuracies = []
        self.w1 = np.random.uniform(-1, 1, size=(self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(-2, 0, size=(1, self.hidden_size))
        self.w2 = np.random.uniform(-1, 1, size=(self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(-2, 0, size=(1, self.output_size))

    def train(self, batch_size):
        start_time = datetime.datetime.now()
        for epoch in range(300):
            batch_x = []
            batch_d = []
            for j in range(1, 500):
                for i in range(1, 13):
                    flat_array = image_to_vector(i, j).flatten()
                    x = np.array(flat_array).T.reshape(1, self.input_size)
                    d = np.zeros((1, self.output_size))
                    d[0][i - 1] = 1
                    batch_x.append(x)
                    batch_d.append(d)

                    if len(batch_x) == batch_size:
                        batch_x = np.vstack(batch_x)
                        batch_d = np.vstack(batch_d)
                        # Forward pass
                        h = batch_x @ self.w1 + self.b1
                        s = sigmoid(h)
                        z = s @ self.w2 + self.b2
                        y = softmax(z)

                        # Backward pass
                        e = y - batch_d
                        d_w2 = s.T @ e
                        d_b2 = np.sum(e, axis=0, keepdims=True)
                        d_w1 = batch_x.T @ (e @ self.w2.T * sigmoid_derivative(s))
                        d_b = np.sum(e @ self.w2.T * sigmoid_derivative(s), axis=0, keepdims=True)

                        # Update parameters
                        self.b2 -= self.learning_rate * d_b2 / batch_size
                        self.w2 -= self.learning_rate * d_w2 / batch_size
                        self.b1 -= self.learning_rate * d_b / batch_size
                        self.w1 -= self.learning_rate * d_w1 / batch_size

                        batch_x = []
                        batch_d = []

            print("Epoch:", epoch, "train_loss:", cross_entropy(d, y), end=" ")
            self.losses.append(cross_entropy(d, y))
            self.training_test()
        end_time = datetime.datetime.now()

        plt.plot(self.losses)
        plt.plot(self.test_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.legend(['Train', 'Test'])
        plt.savefig("weights/loss.png")
        plt.plot(self.test_accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig("accuracy.png")
        print("训练开始时间：", start_time)
        print("训练结束时间：", end_time)

    def training_test(self):
        total = 0
        correct = 0
        for a in range(510, 600):
            for b in range(1, 13):
                flat_array = image_to_vector(b, a).flatten()
                x_test = np.array(flat_array).T.reshape(1, self.input_size)
                d_test = np.zeros((1, self.output_size))
                d_test[0][b - 1] = 1
                h_test = x_test @ self.w1 + self.b1
                s_test = sigmoid(h_test)
                z_test = s_test @ self.w2 + self.b2
                y_test = softmax(z_test)
                index = np.argmax(y_test)
                if index == b - 1:
                    correct += 1
                total += 1
        print("test loss:", cross_entropy(d_test, y_test), "test accuracy:", correct / total)
        self.test_losses.append(cross_entropy(d_test, y_test))
        self.test_accuracies.append(correct / total)

    def SetWeight(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2


if __name__ == '__main__':
    np.random.seed(52)
    bp = BPnet()
    bp.train(2)