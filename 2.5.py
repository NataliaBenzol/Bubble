import cv2,numpy as np
import os 
from pathlib import Path
import shutil
from torchvision import transforms
from scipy.ndimage import gaussian_filter, map_coordinates

def is_black_image(img,threshold=10,min_nonblack_ratio=0.02):
    if img is None:return True
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return(np.sum(gray>threshold)/gray.size)<min_nonblack_ratio

def has_bubbles(img,bubble_threshold=80,min_bubble_area=50,min_bubble_count=3):
    if img is None:return False
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    margin=5
    if h<=2*margin or w<=2*margin:return False
    gray_cropped=gray[margin:h-margin,margin:w-margin]
    _,binary=cv2.threshold(gray_cropped,bubble_threshold,255,cv2.THRESH_BINARY_INV)
    kernel=np.ones((3,3),np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=1)
    binary=cv2.morphologyEx(binary,cv2.MORPH_DILATE,kernel,iterations=1)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len([cnt for cnt in contours if cv2.contourArea(cnt)>min_bubble_area])>=min_bubble_count
    
def filter_images(input_folder,output_folder=None,black_threshold=10,min_nonblack_ratio=0.02,bubble_threshold=80,min_bubble_area=50,min_bubble_count=3,delete_originals=False):
    if output_folder:os.makedirs(output_folder,exist_ok=True)
    extensions=('.jpg','.jpeg','.png','.bmp','.tiff')
    image_paths=set()
    input_path=Path(input_folder)
    for ext in extensions:
        image_paths.update(input_path.rglob(f'*{ext}'))
        image_paths.update(input_path.rglob(f'*{ext.upper()}'))
    kept=0
    removed=0
    for img_path in image_paths:
        img=cv2.imread(str(img_path))
        if is_black_image(img,black_threshold,min_nonblack_ratio)or not has_bubbles(img,bubble_threshold,min_bubble_area,min_bubble_count):
            if delete_originals and output_folder is None:os.remove(img_path)
            removed+=1
            continue
        if output_folder:
            save_path=Path(output_folder)/img_path.relative_to(input_path)
            save_path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(img_path,save_path)
        kept+=1
    print(f"Всего:{kept+removed}\nСохранено:{kept}\nУдалено:{removed}")
    return kept,removed
if __name__=="__main__":
    filter_images("resized_frames","filtered_frames",10,0.02,80,50,3,False)
train_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.3),transforms.RandomRotation(degrees=15),transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),transforms.RandomResizedCrop(224,scale=(0.9,1.0)),transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
