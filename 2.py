import cv2
import os
import numpy as np
from pathlib import Path
import argparse

def resize_with_padding(img, target_size=(224, 224), padding_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    return canvas

def resize_images(input_folder, output_size=(224, 224), output_folder=None, 
                  keep_aspect_ratio=True, padding_color=(0, 0, 0)):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(input_folder).rglob(f'*{ext}'))
        image_paths.extend(Path(input_folder).rglob(f'*{ext.upper()}'))
    resized_count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f" не читает: {img_path}")
                continue
            if keep_aspect_ratio:
                img_resized = resize_with_padding(
                    img, 
                    target_size=output_size, 
                    padding_color=padding_color
                )
            else:
                img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
            if output_folder:
                relative_path = img_path.relative_to(input_folder)
                save_path = Path(output_folder) / relative_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = img_path
            cv2.imwrite(str(save_path), img_resized)
            resized_count += 1
            if resized_count % 100 == 0:
                print(f"обработано: {resized_count}/{len(image_paths)}")
        except Exception as e:
            print(f" ошибка  {img_path}: {e}") 
    return resized_count
if __name__ == "__main__":
    resize_images(
        input_folder="output_frames",      
        output_folder="resized_frames",    
        output_size=(224, 224),           
        keep_aspect_ratio=True,            
        padding_color=(0, 0, 0)            
    )