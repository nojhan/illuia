from random import seed
import numpy as np
from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit


########################################################################
# General parameters
########################################################################

# xmin = -5.12
# xmax =  5.12
xmin = -3
xmax =  5
hres = 50
lres = 15
cnb = 20
samp_big_n = 100
gap = 50
ceil = 150
view_alt = 32
view_angle = 105
#nb_frames = 100
nb_frames = 10

print(nb_frames,"frames")


########################################################################
# Plot initialization
########################################################################

np.random.seed(0)

fig = plt.figure(facecolor='black')
ax = fig.gca(projection='3d', axisbg="black")
ax.view_init(view_alt, view_angle)
# ax.axis('off')

# Black grid background
ax.w_xaxis.set_pane_color((0,0,0))
ax.w_yaxis.set_pane_color((0,0,0))
ax.w_zaxis.set_pane_color((0,0,0))

# Transparent axis grid lines
ax.w_xaxis._axinfo.update({'grid' : {'color': (1,1,1, 0.2)}})
ax.w_yaxis._axinfo.update({'grid' : {'color': (1,1,1, 0.2)}})
ax.w_zaxis._axinfo.update({'grid' : {'color': (1,1,1, 0.2)}})

plt.hold(True)


########################################################################
# Data grids and samples
########################################################################

# Low res grid
glx = np.arange(xmin, xmax, (xmax-xmin)/lres)
gly = np.arange(xmin, xmax, (xmax-xmin)/lres)
grid_low = np.meshgrid(glx, gly)

# Big res grid
ghx = np.arange(xmin, xmax, (xmax-xmin)/hres)
ghy = np.arange(xmin, xmax, (xmax-xmin)/hres)
grid_big = np.meshgrid(ghx, ghy)

# Big res sample
samp_big_x = np.random.random(samp_big_n)*(xmax-xmin)+xmin
samp_big_y = np.random.random(samp_big_n)*(xmax-xmin)+xmin
samp_big = (samp_big_x, samp_big_y)


########################################################################
# Fitness values
########################################################################

# "Real" problem: Rastrigin function
def rastrigin(x,y):
    a=10
    samp_big_n=2
    return a*samp_big_n + (x*x-a*np.cos(2*np.pi*x)) + (y*y-a*np.cos(2*np.pi*y))

# "Model": a quadratic function
def sphere(x, a, b):
    return a * (x[0]*x[0] + x[1]*x[1]) + b

def gauss_full(x, sx, sy, amp, b, x0, y0, ang):
    a = np.cos(ang)*np.cos(ang)/2*sx*sx + np.sin(ang)*np.sin(ang)/2*sy*sy
    b = -1*np.sin(2*ang)/4*sx*sx + np.sin(2*ang)/4*sy*sy
    c = np.sin(ang)*np.sin(ang)/2*sx*sx + np.cos(ang)*np.cos(ang)/2*sy*sy
    return amp * np.exp(-1*(a*(x[0]-x0)*(x[0]-x0) + 2*b*(x[0]-x0)*(x[1]-y0) + c*(x[1]-y0)*(x[1]-y0))) + b

def gauss_sb(x, std, b):
    return -1e2 * np.exp(-1*(x[0]*x[0]/2*std*std + x[1]*x[1]/2*std*std)) + b

def gauss_sabx(x, std, amp, b, x0):
    return -1*amp * np.exp(-1*((x[0]-x0[0])*(x[0]-x0[0])/2*std*std + (x[1]-x0[1])*(x[1]-x0[1])/2*std*std)) + b


# Sampled Rastrigin
samp_big_rast = rastrigin(*samp_big)
grid_big_rast = rastrigin(*grid_big)
grid_low_rast = rastrigin(*grid_low)

# Fit the quadratic model to the sample
sopt, scov = curve_fit(sphere, samp_big, samp_big_rast)
grid_big_sphr = sphere(grid_big, *sopt)

# Fit the gaussian model to the sample
gbounds = (0,[1,1e3])
guess = [0.1,1e2]
g0opt, g0cov = curve_fit(gauss_sb, samp_big, samp_big_rast, p0=guess, bounds=gbounds)
print(g0opt)
grid_big_gaus = gauss_sb(grid_big, *g0opt) 

side=(xmax-xmin)/7
samp_big_zoom_x = np.random.random(samp_big_n)*side-side/2
samp_big_zoom_y = np.random.random(samp_big_n)*side-side/2
samp_big_zoom = (samp_big_zoom_x, samp_big_zoom_y)
samp_big_zoom_rast = rastrigin(*samp_big_zoom)

# Fit the gaussian model to the zoomed sample
gbounds = (0,[1e5,1e5])
guess = [1,1e2]
g1opt, g1cov = curve_fit(gauss_sb, samp_big_zoom, samp_big_zoom_rast, p0=guess, bounds=gbounds)
print(g1opt)
grid_big_gaus_zoom = gauss_sb(grid_big, *g1opt)


########################################################################
# Plot set up
########################################################################

# Plots that have their z-values manipulated during the animation
common_props = {"markeredgecolor":"none", "alpha":0.7}
first_sample = None
second_sample = None

def init():
    # Grid Rastrigin surface
    ax.plot_surface(*grid_big, grid_big_rast, alpha=0.2, color="magenta");

    # Ground contour
    ax.contour(*grid_low, grid_low_rast, offset=-1, colors="green")

    # Optimum star
    ax.scatter([0], [0], [gap], c="red", s=200, edgecolors="yellow", marker='*')

    # Ground contour around optimum
    ax.contour(*grid_big, grid_big_sphr, offset=-1, colors="red", levels=np.arange(0,1,0.2))

    # Model 3D contour
    #ls = range(0,int(np.ceil(np.max(grid_big_sphr))) +gap,int(np.ceil((np.max(grid_big_sphr) +gap)/cnb)))
    #ax.contour(*grid_big, grid_big_sphr +gap, colors="#ffcc00",levels=ls)

    lg0 = range(0,int(np.ceil(np.max(grid_big_gaus))) +gap,int(np.ceil((np.max(grid_big_gaus) +gap)/cnb)))
    ax.contour(*grid_big, grid_big_gaus +gap, colors="#ffcc00",levels=lg0)

    lg1 = range(0,int(np.ceil(np.max(grid_big_gaus_zoom))) +gap,int(np.ceil((np.max(grid_big_gaus_zoom) +gap)/cnb)))
    ax.contour(*grid_big, grid_big_gaus_zoom +gap, colors="#ff5500",levels=lg1)


def sample_down(points,x,z,i,istart,iend):
    n = i - istart
    nmax = iend - istart
    x,y = x
    for j,p in enumerate(points):
        p.set_data(x[j], y[j])
        zr = ceil - (z[j]+gap)
        zi = 1 - n/nmax
        zj = z[j] + gap + zi * zr
        p.set_3d_properties(zj)

    return points


def first_sample_down(artists, i, istart, iend):
    #ax.scatter(*samp_big, samp_big_rast +gap, c="white",     s=30, edgecolors="none")
    return sample_down(artists, samp_big, samp_big_rast, i, istart, iend)

def second_sample_down(artists, i, istart, iend):
    #ax.scatter(*samp_big_zoom, samp_big_zoom_rast +gap, c="#8888ff", s=30, edgecolors="none")
    return sample_down(artists, samp_big_zoom, samp_big_zoom_rast, i, istart, iend)


# animation function.  This will be called sequentially with the frame number
def animate(i):
    global first_sample, second_sample

    print("{}/{}".format(i,nb_frames), flush=True)

    if i <= nb_frames//2:
        if first_sample == None:
            first_sample = sum([ax.plot([], [], [], 'o', c="white", **common_props)
                for k in range(samp_big_n)], [])
        artists = first_sample_down(first_sample, i, 0, nb_frames//2)
    else:
        if second_sample == None:
            second_sample = sum([ax.plot([], [], [], 'o', c="#8888ff" , **common_props)
                for k in range(samp_big_n)], [])
        artists = second_sample_down(second_sample, i, nb_frames//2, nb_frames)

    # Rotate the view point around
    #ax.view_init(view_alt, view_angle + np.degrees(i*(2*np.pi/nb_frames)))
    fig.canvas.draw()

    return artists

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nb_frames, interval=30)#, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('illuia.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

anim.save('illuia.gif',writer='imagemagick',fps=20);
#init()

plt.show()

