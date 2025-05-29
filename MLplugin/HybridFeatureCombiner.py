from meshroom.core import desc

class HybridFeatureCombiner(desc.CommandLineNode):
    commandLine = 'hybridFeatureCombiner --inputSfM {inputSfMValue} --inputFeatures {inputFeaturesValue} --inputMatches {inputMatchesValue} --superpointFeatures {superpointFeaturesValue} --superglueMatches {superglueMatchesValue} --describerTypes {describerTypesValue} --outputFeatures {outputFeaturesValue} --outputMatches {outputMatchesValue}'

    category = 'ML Plugin'
    documentation = '''
Combines features and matches from traditional (SIFT) and deep learning (SuperPoint/SuperGlue) approaches.
Creates a unified set of features and matches while removing duplicates.
Outputs separate folders for combined features and matches.
'''
    DESCRIBER_TYPES = ["sift", "sift_float", "sift_upright", "dspsift", "akaze", "akaze_liop", "akaze_mldb", "cctag3",
                   "cctag4", "sift_ocv", "akaze_ocv", "tag16h5", "unknown"]

    inputs = [
        desc.File(
            name="inputSfM",
            label="Input SfMData",
            description="Input SfMData file.",
            value="",
            uid=[0],
        ),
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
        desc.File(
            name="inputMatches",
            label="Original Matches",
            description="Folder containing original match files.",
            value="",
            uid=[0],
        ),
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
        desc.File(
            name="superglueMatches",
            label="SuperGlue Matches",
            description="Folder containing SuperGlue match files.",
            value="",
            uid=[0],
        ),
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

    def processChunk(self, chunk):
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
        
        self.commandLine = self.commandLine.format(**cmd_args)
        super().processChunk(chunk)