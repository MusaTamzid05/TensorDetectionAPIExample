import tensorflow as tf

def create_session(fraction = 0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

