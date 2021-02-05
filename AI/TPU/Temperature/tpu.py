import subprocess


class Tpu:
    def __init__(self):
        self.amount = 0
        self.mean = 0

    def get_Temperature(self):
        self.amount += 1
        temp = subprocess.run(
            ['cat', '/sys/class/apex/apex_0/temp'], stdout=subprocess.PIPE).stdout.decode("utf-8")
        temp = temp.replace("\n", "")
        temp = int(temp)
        self.mean = (self.mean * (self.amount - 1) /
                     self.amount) + temp/self.amount

        return (temp, self.mean)
