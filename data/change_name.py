import os

def change_name():
    root_dir = 'real-data/'
    folders = [files for _, files, _ in os.walk(root_dir)][0]
    os.chdir(root_dir)
    
    for folder in folders:
        for _, _, files in os.walk(folder):
            for i, file in enumerate(files):
                if os.path.exists(os.path.join(folder, folder+str(i+1)+'.jpg')):
                    continue
                os.rename(os.path.join(folder, files[i]), os.path.join(folder, folder+str(i+1)+'.jpg'))
def main():
    change_name()

if __name__ == '__main__':
	main()