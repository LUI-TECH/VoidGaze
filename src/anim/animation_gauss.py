import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

fig, ax = plt.subplots(figsize = (8,8))
xdata, ydata = [], []

ln, = ax.plot([], [], 'ro', animated=False,label = 'Ground truth')
ln2, = ax.plot([], [], 'bo', animated=False,label = 'Predicted gaze position')
distribution = Ellipse((0.5*3840, 0.5*2048), width=0, height=0,alpha = 0.4,color = 'yellow',label = 'Cone of vision')
ax.add_patch(distribution)

fov = Ellipse((0.5*3840, 0.5*2048), width=0.125*3840, height=0.25*2048,alpha = 0.4,color = 'black',label = 'Field of vision')
ax.add_patch(fov)

target= np.load("/Users/louitech_zero/VoidGaze/Data/results/target.npy") #target
std = np.load("/Users/louitech_zero/VoidGaze/Data/results/gaussstd.npy") #gaussstd
mean = np.load("/Users/louitech_zero/VoidGaze/Data/results/gaussmean.npy") # gaussmean
print(mean.shape,std.shape) 
diff = abs(target-mean)
diffx = np.mean(diff[:,0])
diffy = np.mean(diff[:,1])

Tx = []
Ty = []
Mx = []
My = []
Range = [3000,4000]
def init():
    ax.set_xlim(1400, 2400)
    ax.set_ylim(300, 1700)
    return ln,ln2,distribution


def update(frame):
    frame = int(frame)
    tx = target[frame,0]*3840
    ty = target[frame,1]*2048
    mx = mean[frame,0]*3840
    my = mean[frame,1]*2048
    stdx = std[frame]*3840#max(std[frame,0]*3840*0.4,diffx*3840)     #std[frame]*3840
    stdy = std[frame]*2048#max(std[frame,1]*2048*0.4,diffy*2048)    #max(std[frame,1]*2048,diffy*2048) # stdx

    Tx.append(tx)
    Ty.append(ty)
    Mx.append(mx)
    My.append(my)

    ln.set_data(Tx, Ty)
    ln2.set_data(Mx, My)



    distribution.set_center(xy=(mx, my))
    distribution.set_height(stdx*2*1.645)
    distribution.set_width(stdy*2*1.645)

    return ln,ln2,distribution

ani = FuncAnimation(fig, update, frames=np.linspace(Range[0], Range[1]-Range[0]+1, Range[1]),
                    init_func=init, blit=True)
plt.legend()
plt.xlabel('Horizontal /pixel')
plt.ylabel('Vertical /pixel')
plt.show()