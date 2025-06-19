from meshroom.core import desc
import os

class SuperGlueFeatureMatching(desc.CommandLineNode):
    # Command line template for executing the SuperGlue feature matching script
    commandLine = (
        'superGlue_featureMatching '
        '--input {inputValue} '
        '--pairs {imagePairsValue} '
        '--features {featuresValue} '
        '--weights {weightsValue} '
        '--weightsType {weightsType} '
        '--matchThreshold {matchingThresholdValue} '
        '--sinkhornIterations {sinkhornIterationsValue} '
        '--describerTypes {describerTypesValue} '
        '--ransacThreshold {ransacThresholdValue} '
        '--ransacMaxTrials {ransacMaxTrialsValue} '
        '--output {outputValue} '
        '{forceCpuFlag}'
    )

    # Category shown in Meshroom UI
    category = 'ML Plugin'  

    # Description shown in the Meshroom UI and documentation
    documentation = '''SuperGlue Feature Matcher for Meshroom.

This module integrates the SuperGlue deep learning-based feature matching algorithm into the Meshroom photogrammetry pipeline. 
SuperGlue performs context-aware matching between sets of keypoints extracted from input images, allowing for more robust and 
accurate correspondence estimation compared to traditional methods.

It supports both indoor and outdoor model variants and allows users to configure parameters such as matching thresholds, 
Sinkhorn iterations, and CPU/GPU execution. The module requires precomputed image features (e.g., from SuperPoint) and an 
image pair list to define the matching scope.

Before execution, make sure the required SuperGlue weights are available in the specified directory.'''

    # Paths to pretrained model weights
    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "data")
    WEIGHTS_INDOOR_TEMP = os.path.join(WEIGHTS_DIR, "superglue_indoor.pth")
    WEIGHTS_OUTDOOR_TEMP = os.path.join(WEIGHTS_DIR, "superglue_outdoor.pth")
    WEIGHTS_INDOOR = WEIGHTS_INDOOR_TEMP.replace("\\", "/")
    WEIGHTS_OUTDOOR = WEIGHTS_OUTDOOR_TEMP.replace("\\", "/")

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

        # Text file listing the image pairs to be matched 
        desc.File(
            name="imagePairs",
            label="Image Pairs",
            description="Text file with pairs to match.",
            value="",
            uid=[0],
        ),

        # List of folders where image features are stored
        desc.ListAttribute(
            elementDesc=desc.File(
                name="featuresFolder",
                label="Feature Folder",
                description="Folder containing extracted features.",
                value="",
                uid=[0],
            ),
            name="featuresFolders",
            label="Features Folders",
            description="Folders containing extracted features.",
            group="",
        ),

        # Selection of pretrained SuperGlue weights: 'indoor' or 'outdoor'
        desc.ChoiceParam(
            name="weightsChoice",
            label="Weights Type",
            description="SuperGlue pretrained weights",
            values=["indoor", "outdoor"],
            value="indoor",
            exclusive=True,
            uid=[1],
        ),

        # Confidence threshold for filtering weak matches (0.0–1.0)
        desc.FloatParam(
            name="matchingThreshold",
            label="Match Threshold",
            description="Minimum confidence threshold for matches (0-1)",
            value=0.5,
            range=(0.0, 1.0, 0.01),
            uid=[1],
        ),

        # Number of refinement steps using the Sinkhorn algorithm
        desc.IntParam(
            name="sinkhornIterations",
            label="Sinkhorn Iterations",
            description="Number of matching refinement iterations",
            value=20,
            range=(1, 100, 1),
            uid=[1],
        ),

        # Types of describers used for feature representation
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Output feature format",
            values=["dspsift", "sift"], 
            value=["dspsift"],
            exclusive=False, 
            uid=[1],
        ),

        # Force the matcher to run on CPU even if GPU is available
        desc.BoolParam(
            name="forceCpu",
            label="Force CPU",
            description="Disable GPU acceleration",
            value=False,
            uid=[1],
        ),
        # RANSAC inlier threshold in pixels
        desc.FloatParam(
            name="ransacThreshold",
            label="RANSAC Threshold",
            description="RANSAC inlier threshold (pixels)",
            value=1.5,
            range=(0.1, 10.0, 0.1),
            uid=[1],
        ),

        # Maximum number of RANSAC iterations
        desc.IntParam(
            name="ransacMaxTrials",
            label="RANSAC Max Trials",
            description="Maximum number of RANSAC iterations",
            value=1000,
            range=(100, 10000, 100),
            uid=[1],
        ),
    ]


    # Output folder for resulting match files
    outputs = [
        desc.File(
            name="output",
            label="Matches Folder",
            description="Output directory for match files",
            value=desc.Node.internalFolder,
            uid=[],
        ),
    ]

    def __init__(self):
        super().__init__()
        # Ensure weights directory exists and contains required files
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        if not all(os.path.exists(p) for p in [self.WEIGHTS_INDOOR, self.WEIGHTS_OUTDOOR]):
            raise FileNotFoundError("SuperGlue weights not found in data directory")

    # Called to run the command for a chunk of input data
    def processChunk(self, chunk):
        # Convert the list of features folders into a space-separated string
        features_folders = ' '.join(f'"{f.value}"' for f in chunk.node.featuresFolders.value if f.value)

        # Create dictionary of arguments to substitute into commandLine string
        cmd_args = {
            'inputValue': chunk.node.input.value,
            'imagePairsValue': chunk.node.imagePairs.value,
            'featuresValue': features_folders,
            'weightsValue': self.WEIGHTS_INDOOR if chunk.node.weightsChoice.value == "indoor" else self.WEIGHTS_OUTDOOR,
            'weightsType': chunk.node.weightsChoice.value,
            'matchingThresholdValue': chunk.node.matchingThreshold.value,
            'sinkhornIterationsValue': chunk.node.sinkhornIterations.value,
            'describerTypesValue': ' '.join(f for f in chunk.node.describerTypes.value),
            'outputValue': chunk.node.output.value,
            'forceCpuFlag': ' --forceCpu' if chunk.node.forceCpu.value else '',
            'ransacThresholdValue': chunk.node.ransacThreshold.value, 
            'ransacMaxTrialsValue': chunk.node.ransacMaxTrials.value
        }

        # Fill in the command line with actual values
        self.commandLine = self.commandLine.format(**cmd_args)

        # Call the base class implementatio
        super().processChunk(chunk)
