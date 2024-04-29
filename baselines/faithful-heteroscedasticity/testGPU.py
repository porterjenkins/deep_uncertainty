import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPUs are detected.")
else:
    print(f"GPUs detected: {gpus}")