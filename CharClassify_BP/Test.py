import  numpy as np
import CharClassify_BP.CharClassify_Cross_L2 as CharClassify

bp = CharClassify.BPnet()
correct = 0
total = 0
bp.SetWeight(np.load('weights/w1.npy'), np.load('weights/b1.npy'), np.load(
    'weights/w2.npy'), np.load('weights/b2.npy'))
for a in range(501, 620):
    for b in range(1, 13):
        flat_array = CharClassify.image_to_vector(b, a).flatten()
        x_test = np.array(flat_array).T.reshape(1, bp.input_size)
        d_test = np.zeros((1, bp.output_size))
        d_test[0][b - 1] = 1
        h_test = x_test @ bp.w1 + bp.b1
        s_test = CharClassify.sigmoid(h_test)
        z_test = s_test @ bp.w2 + bp.b2
        y_test = CharClassify.softmax(z_test)
        index = np.argmax(y_test)
        total += 1
        if index == b - 1:
            correct += 1
        print(CharClassify.recognize(index+1), end=" ")
    print()
print("正确率：", (100*correct)/total, "%")