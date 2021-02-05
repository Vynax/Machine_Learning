import threading
import subprocess
import time
import datetime
import status

#global finish


def job(name):
    # for i in range(5):
    starttime = datetime.datetime.now()
    f = open(name, "w")
    print("start time:" + starttime.strftime("%H:%M:%S"))
    f.write("start time:" + starttime.strftime("%H:%M:%S") + "\n")
    amount = 1
    temp = get_Temperature()
    f.write(temp + "\n")
    print(temp)
    mean = int(temp)
    print("mean:" + str(mean))
    while(not status.tpu_finish):
        # print("Child thread:", i)
        amount += 1
        temp = subprocess.run(
            ['cat', '/sys/class/apex/apex_0/temp'], stdout=subprocess.PIPE).stdout.decode("utf-8")
        temp = int(temp.replace("\n", ""))
        print(temp)
        f.write(str(temp) + "\n")
        mean = (mean * (amount - 1) / amount) + temp / amount
        print("mean:" + str(mean))
        time.sleep(0.5)

    finishtime = datetime.datetime.now()
    f.write("mean:" + str(mean) + "\n")
    print("finish time:" + finishtime.strftime("%H:%M:%S"))
    f.write("start time:" + starttime.strftime("%H:%M:%S") + "\n")
    f.write("finish time:" + finishtime.strftime("%H:%M:%S") + "\n")
    print("delta:" + str(finishtime - starttime))
    f.write("delta:" + str(finishtime - starttime) + "\n")
    f.close()


def main():
    #global finish
    finish = False
    t = threading.Thread(target=job, args=("PCI_E_tpu_temp1", finish))
    t.start()
    for i in range(3):
        print("Main thread:", i)
        time.sleep(1)
    finish = True
    t.join()
    print("Done.")


if __name__ == '__main__':
    main()
