from meshroom.core import desc
import os

class SuperPointFeatureExtraction(desc.CommandLineNode):
    # Command line template for executing the SuperPoint feature extraction script
    commandLine = (
        'superPoint_featureExtraction '
        '--input {inputValue} '
        '--weights {weightsValue} '
        '--maxKeypoints {maxKeypointsValue} '
        '--nmsRadius {nmsRadiusValue} '
        '--describerTypes {describerTypesValue} '
        '--output {outputValue}'
    )

    # Category shown in Meshroom UI
    category = 'ML Plugin'

    # Description shown in the Meshroom UI and documentation
    documentation = '''Deep Learning-Based Feature Extraction Using SuperPoint.

This module integrates the SuperPoint neural network into the Meshroom photogrammetry workflow to perform keypoint detection and 
descriptor extraction on input images. SuperPoint is a self-supervised deep learning model designed to identify salient interest 
points and generate robust feature descriptors, enabling reliable matching in downstream tasks such as structure-from-motion (SfM) 
and visual SLAM.

Users can configure options such as the maximum number of keypoints to extract, the model weights to use, and the types of descriptors 
to generate. The extracted features are saved in a format compatible with subsequent modules, such as SuperGlue for feature matching.

Ensure that the SuperPoint weights are correctly specified and available before running the module.'''

    # Default path to the pretrained SuperPoint weights
    WEIGHTS_PATH_TEMP = os.path.join(os.path.dirname(__file__), "data", "superpoint_v1.pth")
    WEIGHTS_PATH = WEIGHTS_PATH_TEMP.replace("\\", "/")

    # Input parameters for the node
    inputs = [
        # Input SfMData file containing camera intrinsics/extrinsics and image list
        desc.File(
            name="input",
            label="SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),

        # Path to the pretrained SuperPoint weights (.pth format)
        desc.File(
            name="weights",
            label="SuperPoint Weights",
            description="Path to SuperPoint weights file (.pth).",
            value=WEIGHTS_PATH,
            uid=[1], 
        ),

        # Maximum number of keypoints to detect per image
        desc.IntParam(
            name="maxKeypoints",
            label="Max Keypoints",
            description="Maximum number of keypoints to detect (-1 for no limit).",
            value=1000,
            range=(-1, 10000, 100),
            uid=[1],
        ),

        # Neighborhood radius for Non-Maximum Suppression or distance filtering
        desc.IntParam(
            name="nmsRadius",
            label="NMS Radius",
            description="Neighborhood radius for Non-Maximum Suppression or distance filtering.",
            value=4,  # Default value; adjust as needed
            range=(0, 50, 1),
            uid=[1],
        ),

        # Choice of descriptor type (should match the one used in matching stage)
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Output feature format",
            values=["dspsift", "sift"], 
            value=["dspsift"],
            exclusive=False,  # Allow multiple values if needed
            uid=[1],
        ),
    ]

    # Define the outputs of the node
    outputs = [
        # Output folder where the features and descriptors will be saved
        desc.File(
            name="output",
            label="Features Folder",
            description="Output path for the features and descriptors files.",
            value=desc.Node.internalFolder,  # Auto-generated internal path
            uid=[],
        ),
    ]

    def __init__(self):
        super().__init__()
        # Validate the existence of the SuperPoint weights file when initializing the node
        if not os.path.exists(self.WEIGHTS_PATH):
            raise FileNotFoundError(f"SuperPoint weights not found at {self.WEIGHTS_PATH}")

    # Called for processing each chunk (e.g., per-image)
    def processChunk(self, chunk):
        # Prepare the actual values to substitute in the command line template
        cmd_args = {
            'inputValue': chunk.node.input.value,
            'weightsValue': chunk.node.weights.value,
            'maxKeypointsValue': chunk.node.maxKeypoints.value,
            'nmsRadiusValue': chunk.node.nmsRadius.value,
            'describerTypesValue': ','.join(chunk.node.describerTypes.value),
            'outputValue': chunk.node.output.value
        }

        # Format the command line string with the actual arguments
        self.commandLine = self.commandLine.format(**cmd_args)

        # Call the parent class's processing method to run the command
        super().processChunk(chunk)