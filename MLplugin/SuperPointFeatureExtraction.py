__version__ = "1.0"

from meshroom.core import desc

class SuperPointFeatureExtraction(desc.CommandLineNode):
    commandLine = 'python superpoint_featureExtraction.py --input {inputValue} --output {outputValue} --weights {weightsValue}'

    category = 'ML Plugin'
    documentation = '''
Deep learning-based feature extraction using SuperPoint.
'''

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
            value="",
            uid=[0],
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Features Folder",
            description="Output path for the features and descriptors files (*.feat, *.desc).",
            value=desc.Node.internalFolder,
            uid=[],
        ),
    ]