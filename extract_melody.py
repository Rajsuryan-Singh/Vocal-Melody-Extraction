import numpy as np

from project.MelodyExt import feature_extraction
from project.utils import load_model, save_model, matrix_parser
from project.test import inference


def extract_melody(y, sr, model = "Seg"):

    # Feature extraction
    feature = feature_extraction(y, sr)
    feature = np.transpose(feature[0:4], axes=(2, 1, 0))

    # load model
    model = load_model(model)

    # Inference
    print(feature[:, :, 0].shape)
    extract_result = inference(feature= feature[:, :, 0],
                                model = model,
                                batch_size=10)

    # Output
    r = matrix_parser(extract_result)

    return r


