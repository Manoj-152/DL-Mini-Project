import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


res = unpickle('dataset/cifar-10-python/cifar-10-batches-py/data_batch_2')
# res = unpickle('dataset/cifar-10-python/cifar-10-batches-py/test_batch')
print(res[b'data'].shape)
num = 1

first_pic = res[b'data'][num].reshape(3,32,32).transpose(1,2,0)
print(first_pic.shape)
print(res.keys())
print(res[b'labels'][num])
print(res[b'filenames'][num])
plt.imshow(first_pic)
plt.savefig('scrap.png')