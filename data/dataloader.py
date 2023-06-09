import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
# import mc
import io


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False

class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__()
        self.initialized = False

        prefix = './'
        image_folder_prefix = 'dataset/DUTS/DUTS-TR'
        if mode == 'train':
            image_list = os.path.join(prefix, 'sod_train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'DUTS-TR-Image')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, eval]')


        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split(",")
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug


class ImagenetContrastive(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        # img = self.load_image(filename)
        img = self.rgb_loader(filename)
        return self.transform(img)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class Imagenet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label




