import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import tpu
#import tmp102


# Initialize communication with TMP102
# tmp102.init()

# This function is called periodically from FuncAnimation


def animate(i, xs, ys, ax, starttime, tpu1):
    # Read temperature (Celsius) from TMP102
    # temp_c = random.randint(20, 60)
    temp = tpu1.get_Temperature()
    # round(tmp102.read_temp(), 2)

    # Add x and y to lists
    td = dt.datetime.now()
    td = td - starttime

    txt = str(td.days)+":"+str(int((td.seconds/3600) % 24))+":" + \
        str(int(td.seconds / 60))+":"+str(td.seconds % 60)+":"+"{microsec:.2f}"
    txt = txt.format(microsec=(td.microseconds+5000)/10000)
    xs.append(txt)
    ys.append(temp[0]/1000)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Mean Temperature:' + str(temp[1]))
    plt.ylabel('Temperature (deg C)')


def setup(starttime):
    plt.style.use('ggplot')
    # Create figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []
    # Set up plot to call animate() function periodically
    t1 = tpu.Tpu()
    ani = animation.FuncAnimation(
        fig, animate, fargs=(xs, ys, ax, starttime, t1), interval=500)
    plt.show()


setup(dt.datetime.now())
