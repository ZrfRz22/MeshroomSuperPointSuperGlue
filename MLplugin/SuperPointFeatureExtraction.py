from meshroom.core import desc

class SuperPointFeatureExtraction(desc.CommandLineNode):
    category = "ML Plugin"
    documentation = "A Machine Learning-based Feature Extractor based on the SuperPoint model"

    commandLine = 'superpoint_feature_extraction.exe --input {inputImage} --weights {weightsFile} --output {outputFile}'

    inputs = [
        desc.File(
            name='inputImage',
            label='Input Image',
            description='Path to the input image.',
            value='',
            uid=[0],
        ),
        desc.File(
            name='weightsFile',
            label='SuperPoint Weights File',
            description='Path to the pre-trained SuperPoint weights.',
            value='superpoint_weights.pth',
            uid=[],
        ),
    ]

    outputs = [
        desc.File(
            name='outputFile',
            label='Output Feature File',
            description='File to store extracted features.',
            value='superpoint_features.txt',
            uid=[0],
        ),
    ]
