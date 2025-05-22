__version__ = "1.0"

from meshroom.core import desc
import os

class SuperGlueFeatureMatching(desc.CommandLineNode):
    commandLine = 'superGlue_featureMatching --input "{inputValue}" --pairs "{imagePairsValue}" --features {featuresValue} --output "{outputValue}" --weights "{weightsValue}" --weightsType {weightsType} --matchThreshold {matchingThresholdValue} --sinkhornIterations {sinkhornIterationsValue} --describerTypes {describerTypesValue} {forceCpuFlag}'

    category = 'ML Plugin'
    documentation = '''SuperGlue feature matcher for Meshroom.'''

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
            description="Text file with pairs to match.",
            value="",
            uid=[0],
        ),
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
        desc.ChoiceParam(
            name="weightsChoice",
            label="Weights Type",
            description="SuperGlue pretrained weights",
            values=["indoor", "outdoor"],
            value="indoor",
            exclusive=True,
            uid=[1],
        ),
        desc.FloatParam(
            name="matchingThreshold",
            label="Match Threshold",
            description="Minimum confidence threshold for matches (0-1)",
            value=0.5,
            range=(0.0, 1.0, 0.01),
            uid=[1],
        ),
        desc.IntParam(
            name="sinkhornIterations",
            label="Sinkhorn Iterations",
            description="Number of matching refinement iterations",
            value=20,
            range=(1, 100, 1),
            uid=[1],
        ),
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Output feature format",
            values=["dspsift", "sift"],  # Simplified to supported types
            value=["dspsift"],
            exclusive=False,  # Changed from exclusive=False
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
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        if not all(os.path.exists(p) for p in [self.WEIGHTS_INDOOR, self.WEIGHTS_OUTDOOR]):
            raise FileNotFoundError("SuperGlue weights not found in data directory")

    def processChunk(self, chunk):
        features_folders = ' '.join(f'"{f.value}"' for f in chunk.node.featuresFolders.value if f.value)
        
        cmd_args = {
            'inputValue': chunk.node.input.value,
            'imagePairsValue': chunk.node.imagePairs.value,
            'featuresValue': features_folders,
            'outputValue': chunk.node.output.value,
            'weightsValue': self.WEIGHTS_INDOOR if chunk.node.weightsChoice.value == "indoor" else self.WEIGHTS_OUTDOOR,
            'weightsType': chunk.node.weightsChoice.value,
            'matchingThresholdValue': chunk.node.matchingThreshold.value,
            'sinkhornIterationsValue': chunk.node.sinkhornIterations.value,
            'describerTypesValue': ' '.join(f for f in chunk.node.describerTypes.value),
            'forceCpuFlag': ' --forceCpu' if chunk.node.forceCpu.value else ''
        }
        
        self.commandLine = self.commandLine.format(**cmd_args)
        super().processChunk(chunk)