from meshroom.core import desc

class FeatureVisualizer(desc.CommandLineNode):
    # # Command line template for executing the Feature Visualizer tool
    commandLine = (
        'featureVisualizer '
        '--inputSfM {inputSfMValue} '
        '--inputFeatures {inputFeaturesValue} '
        '--inputMatches {inputMatchesValue} '
    )

    # Category shown in Meshroom UI
    category = 'ML Plugin'

    # Description shown in the Meshroom UI and documentation
    documentation = '''Feature and Matches Visualizer

    Visualizes the key points and matches for each matched image pairs. 
    Users can cycle through each pair of matched key points between two images to verify the accuracy and robustness of the feature extraction and matching process in a convenient manner. 
    This node has no output, and is only meant to visualize the matches in a separate window.
    '''

    # Input parameters for the node
    inputs = [
        # Input SfMData file containing camera intrinsics/extrinsics and image list
        desc.File(
            name="inputSfM",
            label="Input SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),
        # List feature folders
        desc.ListAttribute(
            elementDesc=desc.File(
                name="inputFeature",
                label="Original Feature",
                description="Folder containing extracted features.",
                value="",
                uid=[0],
            ),
            name="inputFeatures",
            label="Original Features",
            description="Folders containing extracted features.",
            group="",
        ),
        # Matches file folder
        desc.File(
            name="inputMatches",
            label="Original Matches",
            description="Folder containing matched features.",
            value="",
            uid=[0],
        ),
    ]

    # Method that runs when a chunk of the pipeline is being processed
    def processChunk(self, chunk):
        # Build a dictionary of command-line arguments using current node parameters
        cmd_args = {
            'inputSfMValue': chunk.node.inputSfM.value,
            'inputFeaturesValue': ' '.join(f'"{f.value}"' for f in chunk.node.inputFeatures.value if f.value),
            'inputMatchesValue': chunk.node.inputMatches.value
        }

        # Fill in the command line with actual values
        self.commandLine = self.commandLine.format(**cmd_args)

        # Call the base class implementation
        super().processChunk(chunk)
