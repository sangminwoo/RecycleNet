import os
import random
import argparse
import numpy as np

import albumentations as A
import cv2

def get_arguments():
    parser = argparse.ArgumentParser(description='Image Augmentation')
    parser.add_argument('--root_dir', default='dataset-resized/', type=str)
    parser.add_argument('--save_dir', default='augmented/', type=str)
    parser.add_argument('--probability', default='low', help='low, mid, high; probability of applying the transform')
    parser.add_argument('--seed', default=1234, type=int, help='seed for randomize')
    return parser.parse_args()

def augment_and_save(aug_list, root_dir, save_dir):
	img_list = []
	folder_list = os.listdir(root_dir)
	folders = [os.path.join(root_dir, folder) for folder in folder_list]
	for folder in folders:
		files = os.listdir(folder)
		img_list.append([os.path.join(folder, file) for file in files])

	for i in range(len(img_list)):
		if not os.path.exists(os.path.join(save_dir, folder_list[i])):
			os.makedirs(os.path.join(save_dir, folder_list[i]))

		for j, img in enumerate(img_list[i]):
			image = cv2.imread(img)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			save_path = os.path.join(save_dir, folder_list[i], folder_list[i] + str(j) + '.jpg')
			for aug in aug_list:
				augmented = aug(image=image)
				aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
				cv2.imwrite(save_path, aug_img)

class augmentation():
	def __init__(self, probability):
		if probability == 'low': # probability of applying the transform
			self.p = 0.1
		if probability == 'mid':
			self.p = 0.3
		if probability == 'high':
			self.p = 0.5

		self.mean=[0.485, 0.456, 0.406]
		self.std=[0.229, 0.224, 0.225]

	def basic(self): # Basic
		aug = A.Compose([
			A.Normalize(mean=self.mean, std=self.std), # Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.
			A.OneOf([
				A.RandomCrop(height=224, width=224, p=1), # Crop a random part of the input.
				A.CenterCrop(height=224, width=224, p=1), # Crop the central part of the input.
				A.RandomSizedCrop(min_max_height=(256,384), height=224, width=224, p=1), # Crop a random part of the input and rescale it to some size.
				], p=1),
			A.RandomBrightness(limit=0.2, p=self.p), # Randomly change brightness of the input image.
			A.RandomContrast(limit=0.2, p=self.p) # Randomly change contrast of the input image.
			], p=1)

		return aug

	def affine_transform(self): # Affine Transforms: Scale & Translation & Rotation
		aug = A.Compose([
			A.Transpose(p=self.p), # Transpose the input by swapping rows and columns.
			A.OneOf([
				A.RandomRotate90(p=self.p), # Randomly rotate the input by 90 degrees zero or more times.
				A.Rotate(limit=90, p=self.p), # Rotate the input by an angle selected randomly from the uniform distribution.
				A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1 ,rotate_limit=45, p=self.p) # Randomly apply affine transforms: translate, scale and rotate the input.
				], p=1),
			A.OneOf([
				A.HorizontalFlip(p=self.p), # Flip the input vertically around the x-axis.
				A.VerticalFlip(p=self.p), # Flip the input horizontally around the y-axis.
				A.Flip(p=self.p) # Flip the input either horizontally, vertically or both horizontally and vertically.
				], p=1)
			], p=1)

		return aug

	
	def blur_and_distortion(self, kernel_size=(3,3)): # Blur & Distortion
		aug = A.Compose([
			A.OneOf([A.Blur(blur_limit=kernel_size, p=self.p), # Blur the input image using a random-sized kernel.
				A.MotionBlur(blur_limit=kernel_size, p=self.p), # Apply motion blur to the input image using a random-sized kernel.
				A.MedianBlur(blur_limit=kernel_size, p=self.p), # Blur the input image using using a median filter with a random aperture linear size.
				A.GaussianBlur(blur_limit=kernel_size, p=self.p) # Blur the input image using using a Gaussian filter with a random kernel size.
			], p=1),
			A.OneOf([
				A.RandomGamma(gamma_limit=(80,120), p=self.p),
				A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=self.p),
				A.ElasticTransform(p=self.p),
				A.HueSaturationValue(p=self.p), # Randomly change hue, saturation and value of the input image.
				A.RGBShift(p=self.p), # Randomly shift values for each channel of the input RGB image.
				A.ChannelShuffle(p=self.p), # Randomly rearrange channels of the input RGB image.
				A.CLAHE(p=self.p), # Apply Contrast Limited Adaptive Histogram Equalization to the input image.
				A.InvertImg(p=self.p), # Invert the input image by subtracting pixel values from 255.
				], p=1),
			A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=self.p), # Apply gaussian noise to the input image.
			A.RandomShadow(p=self.p) # Simulates shadows for the image
			], p=1)

		return aug

def main():
	args = get_arguments()
	
	ROOT_DIR = args.root_dir
	SAVE_DIR = args.save_dir
	AUG_PROB = args.probability

	if args.seed is not None:
		random.seed(args.seed)

	aug = augmentation(AUG_PROB)
	aug_list = [aug.basic(), aug.affine_transform(), aug.blur_and_distortion()]

	augment_and_save(aug_list, ROOT_DIR, SAVE_DIR)

if __name__ == '__main__':
	main()