from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

class CityScapes(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size=8, img_size=(224,224), output_size=(224,224),
                 input_img_paths='.', target_img_paths='.', 
                 augmentation=False, true_one_hot=False, model_type=None,
                 demoing = False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_size = output_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.augmentation = augmentation
        self.true_one_hot = true_one_hot
        self.model_type = model_type
        self.demoing = demoing

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def augment(self, batch_img, batch_msk):
        
        from albumentations import Compose
        from albumentations.augmentations.transforms import ColorJitter
        from albumentations.augmentations.transforms import ShiftScaleRotate
        from albumentations.augmentations.transforms import HorizontalFlip, Blur

        transforms = Compose([
            ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=15, always_apply=False, p=0.35),
            ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.2, always_apply=False, p=0.35),
            Blur(always_apply=False, p=0.0),
            HorizontalFlip(p=0.5),
            ])

        aug_batch_img = []
        aug_batch_msk = []
        
        for img, msk in zip(batch_img, batch_msk):
            #data = {"image": img, "mask": msk}
            aug_data = transforms(image=img, mask=msk)
            aug_batch_img.append(aug_data['image'])
            aug_batch_msk.append(aug_data['mask'])

        return(np.array(aug_batch_img), np.array(aug_batch_msk))

    def one_hot(self, indices, depth):
        indices = indices.squeeze()
        ones = np.ones(indices.shape)
        y = np.zeros(indices.shape+(depth,))
        for i in range(depth):
            msk = (indices == i) * ones
            y[...,i] = msk
        return(y)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.output_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.output_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            #y[j] = img

        x_ = x.copy()
        y_ = y.copy()

        if self.augmentation == True:
            x_, y_  = self.augment(x_, y_)
            if self.demoing == True:
                x_aug = x_.copy()
                y_aug = y_.copy()
            x_ = x_.astype('float32')
            y_ = y_.astype('float32')


        if self.true_one_hot == True:
            y_ = self.one_hot(y_, 8)

        if self.model_type == None:
            pass
        elif self.model_type == 'resnet50_pspnet':
            x_ = resnet_preprocess_input(x_)
        elif self.model_type == 'resnet50_unet':
            x_ = resnet_preprocess_input(x_)
        elif self.model_type == 'vgg_unet':
            x_ = vgg_preprocess_input(x_)

        if self.demoing == True:
            return((x,x_aug), (y,y_aug))
        else:
            return(x_, y_)

    def preprocess(self, x_):
        from tensorflow.image import resize
        from tensorflow import constant

        x_ = constant(np.array(x_))
        x_ = resize(x_, self.img_size)

        if self.model_type == None:
            pass
        elif self.model_type == 'resnet50_pspnet':
            x_ = resnet_preprocess_input(x_)
        elif self.model_type == 'resnet50_unet':
            x_ = resnet_preprocess_input(x_)
        elif self.model_type == 'vgg_unet':
            x_ = vgg_preprocess_input(x_)

        return(x_)

