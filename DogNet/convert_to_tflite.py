import tensorflow as tf

saved_model_dir = "efficientb0"
converter = tf.lite.TFliteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("b0_tflite", "wb") as f:
    f.write(tflite_model)
