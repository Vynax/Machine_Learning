import os
import json


def convert(start, end, width, height, target):
    for file_name in os.listdir():
        if not file_name.endswith(".json"):
            continue
        with open(file_name, "r") as f:
            json_obj = json.load(f)
            for i in range(start, end):
                target.write(
                    "C" + str(i + 1) + "/" + os.path.basename(f.name)[:-5] + ".jpg"
                )
                for data in json_obj:
                    xmax = data["views"][i]["xmax"]
                    if xmax == -1:
                        continue
                    xmin = data["views"][i]["xmin"]
                    ymax = data["views"][i]["ymax"]
                    ymin = data["views"][i]["ymin"]
                    if xmax < 0 or xmin < 0 or ymax < 0 or ymin < 0:
                        continue
                    elif xmax > width - 1 or xmin > width - 1:
                        continue
                    elif ymax > height - 1 or ymin > height - 1:
                        continue
                    print(xmax, " ", xmin, " ", ymax, " ", ymin)
                    xcenter = (xmin + xmax) / 2 / width
                    ycenter = (ymin + ymax) / 2 / height
                    yolo_w = (xmax - xmin) / width
                    yolo_h = (ymax - ymin) / height
                    yolo_format = " 0,{:.6f},{:.6f},{:.6f},{:.6f}".format(
                        round(xcenter, 6),
                        round(ycenter, 6),
                        round(yolo_w, 6),
                        round(yolo_h, 6),
                    )
                    # print(yolo_format)
                    # target.write(" ID:" + str(data["personID"]))
                    target.write(yolo_format)

                target.write("\n")


def main():
    train = open("wild_train.txt", "w")
    val = open("wild_val.txt", "w")
    os.chdir(
        "/home/css-wu/CloudStation/Data/Wildtrack_dataset_full/Wildtrack_dataset/annotations_positions/"
    )
    print(os.getcwd())
    w = 1920
    h = 1080
    convert(0, 5, w, h, train)
    train.close()

    convert(6, 7, w, h, val)
    val.close()


if __name__ == "__main__":
    main()
