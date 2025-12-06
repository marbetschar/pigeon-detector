import cv2
import numpy as np

from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
)

class HailoHEFModel():
    def __init__(self, hef_path: str):

        # Setting VDevice params to disable the HailoRT service feature
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = VDevice(params=params)

        # Loading compiled HEFs to device:
        self.hef = HEF(hef_path)
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        # Get the "network groups" (connectivity groups, aka. "different networks") information from the .hef
        configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.target.configure(self.hef, configure_params)

        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

    def predict(self, x):
        assert x.shape == (224, 224, 3)

        batch = np.expand_dims(x, axis=0).astype(np.float32)
        input_data = { self.input_vstream_info.name: batch }

        y = None
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output_data = infer_pipeline.infer(input_data)
                y = output_data[self.output_vstream_info.name][0]

        return y

if __name__ == "__main__":
    model = HailoHEFModel("models/mobilenet_v2.hef")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    image = cv2.imread("dataset/images/2025-07-08_08-22-36.jpg")
    image_height, image_width = image.shape[:2]

    image_resized = cv2.resize(image, [224, 224])
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = ((image_rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    
    pred = model.predict(image_norm)

    confidence, x_norm, y_norm, w_norm, h_norm = pred
    x = x_norm * image_width
    y = y_norm * image_height
    w = w_norm * image_width
    h = h_norm * image_height
    print([confidence, x, y, w, h])
