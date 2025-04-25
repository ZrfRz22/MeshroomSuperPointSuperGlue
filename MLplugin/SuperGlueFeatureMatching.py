__version__ = "1.0"

from meshroom.core import desc
import os
from pathlib import Path

class SuperGlueFeatureMatching(desc.CommandLineNode):
    commandLine = 'superGlue_featureMatching --input "{inputValue}" --pairs "{imagePairsValue}" --features {featuresValue} --output "{outputValue}" --weights "{weightsValue}" --weightsType {weightsType} --matchThreshold {matchingThresholdValue}{forceCpuFlag}'

    category = 'ML Plugin'
    documentation = '''SuperGlue feature matcher for Meshroom.
    Requires precomputed features from SuperPoint or other feature extractor.'''

    # Define paths relative to node file location
    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "data")
    WEIGHTS_INDOOR = os.path.join(WEIGHTS_DIR, "superglue_indoor.pth")
    WEIGHTS_OUTDOOR = os.path.join(WEIGHTS_DIR, "superglue_outdoor.pth")

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),
        desc.File(
            name="imagePairs",
            label="Image Pairs",
            description="Text file with pairs to match (format: 'viewId1 viewId2')",
            value="",
            uid=[0],
        ),
        desc.ListAttribute(
            elementDesc=desc.File(
                name="featuresFolder",
                label="Feature Folder",
                description="Folder containing extracted features and descriptors.",
                value="",
                uid=[0],
            ),
            name="featuresFolders",
            label="Features Folders",
            description="One or more folders containing extracted features and descriptors.",
            group="",
        ),
        desc.ChoiceParam(
            name="weightsChoice",
            label="Weights Type",
            description="SuperGlue pretrained weights",
            values=["indoor", "outdoor"],
            value="indoor",  # Default to indoor
            exclusive=True,
            uid=[1],
            group="",
        ),
        desc.FloatParam(
            name="matchingThreshold",
            label="Match Threshold",
            description="Minimum confidence threshold for matches (0-1)",
            value=0.5,
            range=(0.0, 1.0, 0.01),
            uid=[1],
        ),
        desc.BoolParam(
            name="forceCpu",
            label="Force CPU",
            description="Disable GPU acceleration",
            value=False,
            uid=[1],
        ),
    ]

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
        # Verify weights exist during initialization
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        if not all(os.path.exists(p) for p in [self.WEIGHTS_INDOOR, self.WEIGHTS_OUTDOOR]):
            raise FileNotFoundError(
                f"SuperGlue weights not found in {self.WEIGHTS_DIR}\n"
                "Required files:\n"
                f"- {os.path.basename(self.WEIGHTS_INDOOR)}\n"
                f"- {os.path.basename(self.WEIGHTS_OUTDOOR)}"
            )

    def processChunk(self, chunk):
        # Prepare features folders string (space-separated, quoted paths)
        features_folders = ' '.join(f'"{f.value}"' for f in chunk.node.featuresFolders.value)
        
        # Get weights selection and path
        weights_type = chunk.node.weightsChoice.value
        weights_path = self.WEIGHTS_INDOOR if weights_type == "indoor" else self.WEIGHTS_OUTDOOR
        
        # Prepare command line arguments
        cmd_args = {
            'inputValue': chunk.node.input.value,
            'imagePairsValue': chunk.node.imagePairs.value,
            'featuresValue': features_folders,
            'outputValue': chunk.node.output.value,
            'weightsValue': weights_path,
            'weightsType': weights_type,  # Pass the actual type (indoor/outdoor)
            'matchingThresholdValue': chunk.node.matchingThreshold.value,
            'forceCpuFlag': ' --forceCpu' if chunk.node.forceCpu.value else ''
        }
        
        # Execute command
        self.commandLine = self.commandLine.format(**cmd_args)
        super().processChunk(chunk)