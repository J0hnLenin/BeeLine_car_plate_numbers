import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import pandas as pd


from template import apply_template
from model import LettersPrediction
from deformation import img_deformation


def predicts(INPUT_FOLDER_PATH):
    data_path = Path(INPUT_FOLDER_PATH)


    regions_type = [2, 3]
    model = LettersPrediction()


    result = []
    for p in tqdm(data_path.iterdir()):

        if not p.suffix in [".png", ".jpg", ".jpeg"]:
            continue
        result.append(
            {
                "image_name": p.stem,
                "prediction_region_length_2": "",
                "prediction_region_length_3": ""
            }
        )

        img = cv2.imread(str(p))

        #вызов фунции деформации
        img = img_deformation(img)
        ########################

        img = cv2.resize(img, (512,112))
        contours = cv2.resize(contours, (512,112))

        for region_type in regions_type:

            crops = apply_template(img, region_type)

            lp_number = model.predict_series(crops)
            result[-1][f"prediction_region_length_{region_type}"] = lp_number

    pd.DataFrame(result).to_csv('modelPredict.csv', index=False)
