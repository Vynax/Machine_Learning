from PIL import Image
import os
from pathlib import Path

os.chdir(
    "/home/css-wu/CloudStation/Data/Wildtrack_dataset_full/Wildtrack_dataset/Image_subsets/"
)
img_png = Image.open("C1/00000000.png").convert("RGB")
img_png.save("JPG/00000000.jpg", "jpeg")

pre = "JPG/"
ch = "C"
# Path("JPG/C1").mkdir(parents=True, exist_ok=True)

# print(os.listdir("C1"))
print(os.path.exists(pre))
for i in range(7):
    dir_name = ch + str(i + 1)
    dir_path = pre + dir_name
    print(dir_name)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(dir_name):
        if file.endswith(".png"):
            img_png = Image.open(dir_name + "/" + file).convert("RGB")
            filename = file[:-4]
            img_png.save(dir_path + "/" + filename + ".jpg", "jpeg")
