from pathlib import Path

# 讓coco的資料轉成只有person(0)的部份
def person_filter(source, target):
    print("hi")
    person_all = 0
    for line in source:
        line = line.replace("\n", "")
        line = line.split(" ")
        pic_name = line[0]
        person_count = 0
        i = 1
        print_count = 0
        while i < len(line):
            line_split = line[i].split(",")
            # print(line_split)
            print_count += 1
            # print(i)
            if line_split[0] == "0":
                person_count += 1
                i += 1
            else:
                del line[i]

        # print("Picture_name:" + pic_name)
        # print("person amount:" + str(person_count))
        # print("print amount:" + str(print_count))

        if person_count > 0:
            person_all += person_count
            target.write(line[0])
            for j in range(1, len(line)):
                target.write(" ")
                target.write(line[j])
            target.write("\n")

    print("all:", person_all)


def main():
    coco_val = open(
        str(Path.home()) + "/Documents/tensorflow-yolov4/test/dataset/val2017.txt", "r"
    )
    val = open("coco_val_only_person.txt", "w")
    person_filter(coco_val, val)
    val.close()
    coco_val.close()

    coco_train = open(
        str(Path.home()) + "/Documents/tensorflow-yolov4/test/dataset/train2017.txt",
        "r",
    )
    train = open("coco_train_only_person.txt", "w")
    person_filter(coco_train, train)
    train.close()
    coco_train.close()


if __name__ == "__main__":
    main()
