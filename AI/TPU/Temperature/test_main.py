import threading
import subprocess
import time
import datetime
import status
import tpu_temperature as tpu_temp


def main():
    #global finish
    status.tpu_finish = False
    t = threading.Thread(target=tpu_temp.job,
                         args=["PCI_E_tpu_temp1"])  # or args=("PCI_E_tpu_temp1",)
    t.start()
    for i in range(3):
        print("Main thread:", i)
        time.sleep(1)
    status.tpu_finish = True
    t.join()
    print("Done.")


main()
