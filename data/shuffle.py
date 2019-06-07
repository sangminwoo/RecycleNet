import argparse
import os
import random

def get_arguments():
    parser = argparse.ArgumentParser(description='Shuffle')
    parser.add_argument('--root_dir', type=str, default='dataset-resized/')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.13)
    parser.add_argument('--test', type=float, default=0.17)
    return parser.parse_args()

def split_data(args):
	root_dir = args.root_dir

	filelist = [files for _, _, files in os.walk(root_dir)][1:]

	def label_data(root_dir):
		with open('trash_index.txt', 'w') as f:
			for files in filelist:
				for index, file in enumerate(files):
					f.write(file + ' ' + str(index) + '\n')

	label_data(root_dir)
	train_rate = args.train
	val_rate = args.val
	test_rate = args.test

	with open('train_index.txt', 'w') as train:
		with open('val_index.txt', 'w') as val:
			with open('test_index.txt', 'w') as test:
				for label, files in enumerate(filelist):
					files = [file + ' ' + str(label+1) for file in files]

					random.shuffle(files)
					num_train = int(train_rate * len(files))
					num_val = int(val_rate * len(files))
					
					for train_data in files[:num_train]:
						train.write(train_data + '\n')

					for val_data in files[num_train:num_train + num_val]:
						val.write(val_data + '\n')

					for test_data in files[num_train + num_val:]:
						test.write(test_data + '\n')

def main():
	args = get_arguments()
	split_data(args)

if __name__ == '__main__':
	main()