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
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)

    def predict(self, x):
        input_data = { self.input_vstream_info.name: np.array([x]) }

        y = None
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output_data = infer_pipeline.infer(input_data)
                y = output_data[self.output_vstream_info.name][0]

        return y

if __name__ == "__main__":
    model = HailoHEFModel("models/mobilenet_v2.hef")

    image = cv2.imread("dataset/images/2025-07-08_07-39-50.jpg")
    image_resized = cv2.resize(image, [224, 224])
    y = model.predict(image_resized)
    print(y)