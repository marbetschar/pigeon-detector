import tensorflow as tf

# Load the graph definition
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='tflite_graph.pb',
    input_arrays=['normalized_input_image_tensor'],
    input_shapes={'normalized_input_image_tensor': [1, 300, 300, 3]},
    output_arrays=[
        'TFLite_Detection_PostProcess',
        'TFLite_Detection_PostProcess:1',
        'TFLite_Detection_PostProcess:2',
        'TFLite_Detection_PostProcess:3'
    ]
)

converter.allow_custom_ops = True
converter.inference_type = tf.uint8
converter.quantized_input_stats = {'normalized_input_image_tensor': (128, 127)}  # mean, std

tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved as model.tflite")
