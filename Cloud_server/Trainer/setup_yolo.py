import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

def setup():
    os.makedirs(config.DETECTION_DIR, exist_ok=True)
    for folder in ['train', 'valid']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(config.DETECTION_DIR, folder, sub), exist_ok=True)
            
    content = "train: " + os.path.abspath(os.path.join(config.DETECTION_DIR, 'train', 'images')).replace('\\', '/') + "\n"
    content += "val: " + os.path.abspath(os.path.join(config.DETECTION_DIR, 'valid', 'images')).replace('\\', '/') + "\n\n"
    content += "nc: 1\nnames: ['hand']\n"
    
    with open(os.path.join(config.DETECTION_DIR, 'data.yaml'), 'w') as f:
        f.write(content)
        
    print("Done setting up YOLO directories.")

if __name__ == '__main__':
    setup()
