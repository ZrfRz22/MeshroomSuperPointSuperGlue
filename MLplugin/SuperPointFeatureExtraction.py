__version__ = "1.0"

from meshroom.core import desc
import os

class SuperPointFeatureExtraction(desc.CommandLineNode):
    commandLine = 'superPoint_featureExtraction --input {inputValue} --output {outputValue} --weights {weightsValue} --maxKeypoints {maxKeypointsValue} --describerType {describerTypeValue}'

    category = 'ML Plugin'
    documentation = '''
Deep learning-based feature extraction using SuperPoint.
'''

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "superpoint_v1.pth")

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),
        desc.File(
            name="weights",
            label="SuperPoint Weights",
            description="Path to SuperPoint weights file (.pth).",
            value=WEIGHTS_PATH,
            uid=[1],
            advanced=True,
        ),
        desc.IntParam(
            name="maxKeypoints",
            label="Max Keypoints",
            description="Maximum number of keypoints to detect (-1 for no limit).",
            value=1000,
            range=(-1, 10000, 100),
            uid=[1],
        ),
        desc.ChoiceParam(
            name="describerType",
            label="Describer Type",
            description="Output feature format",
            values=["dspsift"],  # Simplified to supported types
            value="dspsift",
            exclusive=True,  # Changed from exclusive=False
            uid=[1],
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Features Folder",
            description="Output path for the features and descriptors files.",
            value=desc.Node.internalFolder,
            uid=[],
        ),
    ]

    def __init__(self):
        super().__init__()
        # Verify weights exist during initialization
        if not os.path.exists(self.WEIGHTS_PATH):
            raise FileNotFoundError(f"SuperPoint weights not found at {self.WEIGHTS_PATH}")