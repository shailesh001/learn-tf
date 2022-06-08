import tensorflow as tf
print("Tensor Flow Version: {}".format(tf.__version__))

#print("GPU Available: ", tf.test.is_gpu_available())
print("GPU Available: ", tf.config.list_physical_devices('GPU'))


if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print('GPU Available')
else:
    print('No GPU Available')
