import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from PIL import Image
from keras_transforms import random_transform

TRASH_DICT = {
'1' : 'glass',
'2' : 'paper',
'3' : 'cardboard',
'4' : 'plastic',
'5' : 'metal',
'6' : 'trash'
}

class TrashDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform

        self.train_img_file = os.path.join(root_dir, 'one-indexed-files-notrash_train.txt')
        self.val_img_file = os.path.join(root_dir, 'one-indexed-files-notrash_val.txt')
        self.test_img_file = os.path.join(root_dir, 'one-indexed-files-notrash_test.txt')

        img_dirs = [x[0] for x in os.walk(os.path.join(root_dir, 'dataset-resized'))][1:]

        img_dirs_dict = {}
        for img_dir in img_dirs:
            trash_name = img_dir.split('/')[-1]
            img_dirs_dict[trash_name] = img_dir

        self.img_paths = None
        self.img_annos = None
        if mode == 'train':
            self.train_img_paths = []
            self.train_annotations = []
            with open(self.train_img_file, "r") as lines:
                for line in lines:
                    img_name = line.split(' ')[0]
                    trash_idx = line.split(' ')[-1][0]
                    self.train_img_paths.append( os.path.join( img_dirs_dict[TRASH_DICT[trash_idx]], img_name ) )
                    self.train_annotations.append( trash_idx )
            
            assert len(self.train_img_paths) == len(self.train_annotations)
            self.img_paths = self.train_img_paths
            self.img_annos = self.train_annotations

        elif mode == 'val':
            with open(self.val_img_file, "r") as lines:
                self.val_img_paths = []
                self.val_annotations = []
                for line in lines:
                    img_name = line.split(' ')[0]
                    trash_idx = line.split(' ')[-1][0]
                    self.val_img_paths.append( os.path.join( img_dirs_dict[TRASH_DICT[trash_idx]], img_name ) )
                    self.val_annotations.append( trash_idx )

            assert len(self.val_img_paths) == len(self.val_annotations)
            self.img_paths = self.val_img_paths
            self.img_annos = self.val_annotations

        elif mode == 'test':
            self.test_img_paths = []
            self.test_annotations = []
            with open(self.test_img_file, "r") as lines:
                for line in lines:
                    img_name = line.split(' ')[0]
                    trash_idx = line.split(' ')[-1][0]
                    self.test_img_paths.append( os.path.join( img_dirs_dict[TRASH_DICT[trash_idx]], img_name ) )
                    self.test_annotations.append( trash_idx )

            assert len(self.test_img_paths) == len(self.test_annotations)
            self.img_paths = self.test_img_paths
            self.img_annos = self.test_annotations

        self.num_classes = 6
        # import pdb
        # pdb.set_trace()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transform( Image.open( self.img_paths[idx] ).convert('RGB') )
        anno = int(self.img_annos[idx]) - 1

        if self.mode == 'test':
            return img, anno, self.img_paths[idx]
        else:
            return img, anno
def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims).scatter_(1, indexes, 1)
    one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots

if __name__ == '__main__':
    root_dir = 'data/'    

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=MEAN, std=STD)])

    dataset = TrashDataset(root_dir, img_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)


    while True:
        for seq, (img, anno) in enumerate(dataloader):

            anno = torch.from_numpy( np.asarray(anno) )

            import pdb
            pdb.set_trace()
