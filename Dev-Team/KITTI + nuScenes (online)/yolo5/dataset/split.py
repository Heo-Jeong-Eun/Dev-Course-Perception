import glob
from sklearn.model_selection import train_test_split

img_list = sorted(glob.glob("dataset/training/images/*.png"))

train_list, val_list = train_test_split(img_list, test_size=0.2, random_state=2000)

with open("dataset/train.txt", "w") as f:
    f.write("\n".join(train_list) + "\n")

with open("dataset/val.txt", "w") as f:
    f.write("\n".join(val_list) + "\n")