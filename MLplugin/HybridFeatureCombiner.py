from meshroom.core import desc

class HybridFeatureCombiner(desc.CommandLineNode):
    # Command line template for executing the Hybrid Feature Combiner tool
    commandLine = (
        'hybridFeatureCombiner '
        '--inputSfM {inputSfMValue} '
        '--inputFeatures {inputFeaturesValue} '
        '--inputMatches {inputMatchesValue} '
        '--superpointFeatures {superpointFeaturesValue} '
        '--superglueMatches {superglueMatchesValue} '
        '--describerTypes {describerTypesValue} '
        '--outputFeatures {outputFeaturesValue} '
        '--outputMatches {outputMatchesValue}'
    )

    # Category shown in Meshroom UI
    category = 'ML Plugin'

    # Description shown in the Meshroom UI and documentation
    documentation = '''Hybrid Feature Combiner
    
    Combines features and matches from traditional (DSP-SIFT) and deep learning (SuperPoint/SuperGlue) approaches. 
    Creates a unified set of features and matches while removing duplicates. 
    Outputs separate folders for combined features and matches.'''

    # List of valid feature describer types
    DESCRIBER_TYPES = [
        "sift", "sift_float", "sift_upright", "dspsift", "akaze", "akaze_liop", "akaze_mldb",
        "cctag3", "cctag4", "sift_ocv", "akaze_ocv", "tag16h5", "unknown"
    ]

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
        # Folders with extracted features using traditional feature extraction
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
        # Folders with matched features using traditional feature matching 
        desc.File(
            name="inputMatches",
            label="Original Matches",
            description="Folder containing original match files.",
            value="",
            uid=[0],
        ),
        # Folders with extracted features using SuperPoint
        desc.ListAttribute(
            elementDesc=desc.File(
                name="superpointFeature",
                label="SuperPoint Feature",
                description="Folder containing extracted features.",
                value="",
                uid=[0],
            ),
            name="superpointFeatures",
            label="SuperPoint Features",
            description="Folders containing extracted features.",
            group="",
        ),
        # Folders with matched features using traditional SuperGlue
        desc.File(
            name="superglueMatches",
            label="SuperGlue Matches",
            description="Folder containing SuperGlue match files.",
            value="",
            uid=[0],
        ),
        # Describer type for the output (can be multiple)
        desc.ChoiceParam(
            name="describerTypes",
            label="Describer Types",
            description="Output feature format",
            values=DESCRIBER_TYPES,
            value=["dspsift"],
            exclusive=False,
            uid=[1],
        ),
    ]

    # Output folders for the combined features and matches
    outputs = [
        desc.File(
            name="outputFeatures",
            label="Combined Features",
            description="Output folder for combined features.",
            value=desc.Node.internalFolder + "/features",
            uid=[],
        ),
        desc.File(
            name="outputMatches",
            label="Combined Matches",
            description="Output folder for combined matches.",
            value=desc.Node.internalFolder + "/matches",
            uid=[],
        ),
    ]

    # Called to run the command for a chunk of input data
    def processChunk(self, chunk):
        # Create dictionary of arguments to substitute into commandLine string
        cmd_args = {
            'inputSfMValue': chunk.node.inputSfM.value,
            'inputFeaturesValue': ' '.join(f'"{f.value}"' for f in chunk.node.inputFeatures.value if f.value),
            'inputMatchesValue': chunk.node.inputMatches.value,
            'superpointFeaturesValue': ' '.join(f'"{f.value}"' for f in chunk.node.superpointFeatures.value if f.value),
            'superglueMatchesValue': chunk.node.superglueMatches.value,
            'describerTypesValue': ' '.join(f for f in chunk.node.describerTypes.value),
            'outputFeaturesValue': chunk.node.outputFeatures.value,
            'outputMatchesValue': chunk.node.outputMatches.value,
        }

        # Fill in the command line with actual values
        self.commandLine = self.commandLine.format(**cmd_args)

        # Call the base class implementation
        super().processChunk(chunk)
