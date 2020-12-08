import tensorflow as tf
print("Tensor Flow Version: {}".format(tf.__version__))

print("GPU Available: ", tf.test.is_gpu_available())


if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print('GPU Available')
else:
    print('No GPU Available')
