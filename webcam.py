import argparse
import torch
import torch.nn as nn
import os
import resnet
import cv2
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

TRASH_DICT = {
'1' : 'glass',
'2' : 'metal',
'3' : 'paper',
'4' : 'plastic',
'5' : 'metal',
'6' : 'trash'
}

parser = argparse.ArgumentParser(description='RecycleNet webcam inference')
parser.add_argument('--resume', default='save/model_best.pth.tar', type=str)
parser.add_argument('--cuda', default=False, type=bool)
parser.add_argument('--save_dir', default='capture_img.jpg', type=str)

args = parser.parse_args()
    
if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = nn.DataParallel(resnet.resnet18(pretrained=True, num_classes=len(TRASH_DICT)))
checkpoint = torch.load(args.resume, map_location=device)
state_dict = checkpoint['state_dict']

model.load_state_dict(state_dict)
model.eval()

def inference(save_dir):
    frame = Image.open(save_dir)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    img_transforms = transforms.Compose([transforms.CenterCrop(224),                
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=MEAN, std=STD)])

    image_tensor = img_transforms(frame).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    
    softmax = nn.Softmax(dim=1)
    output = model(Variable(image_tensor))
    pred = softmax(output[0].data).numpy()
    trash_idx = str(pred.argmax()+1)
    pred_class, confidence = TRASH_DICT[trash_idx], pred.max()

    return pred_class, confidence
    '''
    print('.....Prediction in progress.....')
    total = len(img_paths)
    correct = 0; number = 1
    f = open('inference_result.txt', 'w')
    f.write('{}\t{}\t{}\t{}\t{}\n'.format("Number", "Class", "Predicted", "Correct", "Confidence"))
    for img_path, annotation in zip(img_paths, annotations):
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        plt.show()
        
        image = Image.open(img_path)
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        
        img_transforms = transforms.Compose([
                                     transforms.CenterCrop(224),                
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=MEAN, std=STD)])

        image_tensor = img_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor.to(device)
        
        softmax = nn.Softmax(dim=1)
        output = model(Variable(image_tensor))
        pred = softmax(output[0].data).numpy()
        trash_idx = str(pred.argmax()+1)
        pred_class, confidence = TRASH_DICT[trash_idx], pred.max()
        
        if pred_class == annotation:
            correct += 1
        
        f.write('{}\t{}\t{}\t{}\t{:.3}\n'.format(number, annotation, pred_class, pred_class==annotation, confidence))
        number += 1
        
        print('>>> Predicted: {} | Label: {}\n'
              '>>> Correct: {} [{}]/[{}]\n'
              '>>> Confidence: {:.3}'
              .format(pred_class, annotation, pred_class == annotation, correct, total, confidence))

    accuracy = correct/total*100
    f.write('accuracy : {}'.format(accuracy))
    f.close()
    print('\n>>> Accuracy: {:.3} [{}]/[{}]'.format(accuracy, correct, total))
    '''

def main(save_dir):
    cam = cv2.VideoCapture(0)

    if cam.isOpened() == False:
        print("Unable to read camera feed")
     
    while True:
        ret, frame = cam.read()
        
        if ret == True:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 32: # Press 'space bar' to capture image
                cv2.imwrite(save_dir, frame)
                pred_class, confidence = inference(save_dir)
                print(f'Prediction: {pred_class}, Confidence: {confidence}')
            if cv2.waitKey(1) == 27: # Press 'esc' to exit
                break
        else:
            break 
     
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    '''
    infer_img_file = os.path.join(args.root_dir, 'inference_index.txt')
    
    img_dirs = [x[0] for x in os.walk(args.root_dir)][1:]
    img_dirs_dict = {}
    for img_dir in img_dirs:
        trash_name = img_dir.split('/')[-1]
        img_dir = img_dir.replace('/', '\\')
        img_dirs_dict[trash_name] = img_dir
    
    infer_img_paths, infer_annos = [], []
    with open(infer_img_file, "r") as lines:
        for line in lines:
            img_name = line.split()[0]
            trash_idx = line.split()[1]
            infer_img_paths.append(os.path.join(img_dirs_dict[TRASH_DICT[trash_idx]], img_name))
            infer_annos.append(TRASH_DICT[trash_idx])

    assert len(infer_img_paths) == len(infer_annos)
    predict_images(infer_img_paths, infer_annos)
    '''
    main(args.save_dir)