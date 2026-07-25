import os
import sys
import argparse

import torch
import torch.nn.functional as F


CATNET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "CAT-Net"
)

sys.path.insert(0, CATNET_PATH)


from lib import models
from lib.config import config, update_config



class CATNetDetector:

    def __init__(self):

        self.model = self.load_model()


    def load_model(self):

        print("Loading CAT-Net...")


        args = argparse.Namespace(
            cfg="experiments/CAT_full.yaml",
            opts=[
                "TEST.MODEL_FILE",
                "output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar",
                "TEST.FLIP_TEST",
                "False",
                "TEST.NUM_SAMPLES",
                "0"
            ]
        )


        update_config(config, args)


        # Create CAT-Net architecture
        model = eval(
            "models." + config.MODEL.NAME + ".get_seg_model"
        )(config)


        checkpoint_path = os.path.join(
            CATNET_PATH,
            "output",
            "splicing_dataset",
            "CAT_full",
            "CAT_full_v2.pth.tar"
        )


        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda"
        )


        model.load_state_dict(
            checkpoint["state_dict"]
        )


        model = model.cuda()
        model.eval()


        print("CAT-Net loaded successfully.")

        return model



    def detect(self, image, qtable):

        """
        image:
            Tensor [B,24,H,W]

        qtable:
            Tensor [B,8,8]

        returns:
            heatmap [H,W]
        """


        with torch.no_grad():

            image = image.cuda()
            qtable = qtable.cuda()


            output = self.model(
                image,
                qtable
            )


            heatmap = F.softmax(
                output,
                dim=1
            )[:,1]


        return heatmap.cpu()