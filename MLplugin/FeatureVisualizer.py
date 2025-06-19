from meshroom.core import desc

class FeatureVisualizer(desc.CommandLineNode):
    # Command line template for executing the feature combiner tool
    commandLine = (
        'featureVisualizer '
        '--inputSfM {inputSfMValue} '
        '--inputFeatures {inputFeaturesValue} '
        '--inputMatches {inputMatchesValue} '
    )

    # Category label for the Meshroom UI
    category = 'ML Plugin'

    # Description shown in the Meshroom UI and documentation
    documentation = '''
    Combines features and matches from traditional (DSP-SIFT) and deep learning (SuperPoint/SuperGlue) approaches.
    Creates a unified set of features and matches while removing duplicates.
    Outputs separate folders for combined features and matches.
    '''

    # Input parameters for the node
    inputs = [
        # Input SfMData file (camera intrinsics, extrinsics, etc.)
        desc.File(
            name="inputSfM",
            label="Input SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),
        # List of original (classical) feature folders
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
        # Original match file folder (from traditional methods)
        desc.File(
            name="inputMatches",
            label="Original Matches",
            description="Folder containing original match files.",
            value="",
            uid=[0],
        ),
    ]

    # Output folders for the combined features and matches
    outputs = []

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

        # Call the base class implementation (which likely runs the command)
        super().processChunk(chunk)
