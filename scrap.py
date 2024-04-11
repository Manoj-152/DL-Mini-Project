import matplotlib.pyplot as plt
import pickle5 as pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1

# res = unpickle('dataset/cifar-10-python/cifar-10-batches-py/data_batch_2')
res = unpickle('dataset/cifar_test_nolabels.pkl')
# res = unpickle('dataset/cifar-10-python/cifar-10-batches-py/test_batch')
print(res[b'data'].shape)
num = 120

first_pic = res[b'data'][num].reshape(3,32,32).transpose(1,2,0)
print(first_pic.shape)
print(res.keys())
print(res[b'ids'][num])
# print(res[b'labels'][num])
# print(res[b'filenames'][num])
plt.imshow(first_pic)
plt.savefig('scrap.png')