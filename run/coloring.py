import matplotlib.pyplot as plt
import cv2, random, os
import torch, numpy as np
from run.infer import color_map

MASK_DIR = './archive/masks'
N = 10
PALETTE = {
    0: (0, 0, 0),        # Sfondo - Nero
    1: (34, 139, 34),    # Auto - Forest Green
    2: (205, 133, 63),   # Ruote - Peru
    3: (255, 215, 0),    # Fanali - Gold
    4: (0, 191, 255)     # Finestrini - Deep Sky Blue
}

def plot(mask):
    plt.figure(figsize=(6, 6)); plt.imshow(mask); plt.axis("off"); plt.show()


def main():
    files = [f for f in os.listdir(MASK_DIR) if f.endswith(('.png'))]
    
    for i in range(N):
        mask = cv2.imread(os.path.join(MASK_DIR, random.choice(files)), cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).long()
        colored = color_map(mask)
        plot(colored)

if __name__ == '__main__':
    main()