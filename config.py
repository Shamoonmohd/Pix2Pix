import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# for colab
# TRAIN_DIR = "/content/drive/MyDrive/Dataset/Pix2Pix/maps/train"
# VAL_DIR = "/content/drive/MyDrive/Dataset/Pix2Pix/maps/val"
# for local
TRAIN_DIR = "maps/train"
VAL_DIR = "maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
## here we are applying various data augmnetation technique for inpput and target
both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5),], additional_targets={"image0": "image"},
)
## here we are applying various data augmentation technique for inpput 
transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
## here we are applying various data augmnetation technique for target
transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
