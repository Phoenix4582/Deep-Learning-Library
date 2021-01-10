import tensorflow as tf

saved_dir = "model"
converter = tf.lite.TFliteConverter.from_saved_model(saved_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
model_lite = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(model_lite)
