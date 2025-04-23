import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls import *
import sys

def updateFace (key, *args):
    CC.changeFace2(key, svar[key].get())
    verts = CC.getVertices()
    polyc.set_verts(verts[CC.geoTriangles])
    canvas.draw()

assert(len(sys.argv) > 4)
CC = CharacterCreator(sys.argv[1:])

root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", root.quit)
root.title("Character Creator Demo")

notebook = ttk.Notebook(root)
notebook.pack(side=tk.BOTTOM, expand = 1, fill = tk.BOTH)

tabs = dict()
svar = dict()

for k,s in CC.geoSliders.items():
    menu = s.menu
    if menu != None:
        if menu not in tabs:
            tabs[menu] = tk.Frame(notebook)
            tabs[menu].pack(expand = True, fill = tk.BOTH)
            notebook.add(tabs[menu], text=menu)
        svar[k] = tk.IntVar(value=s.guiValue, name=k)
        tkslider = tk.Scale(tabs[menu], orient=tk.HORIZONTAL, label=s.label, from_=s.guiMin, to=s.guiMax, variable=svar[k])
        tkslider.pack(expand = False, fill = 'x')
        svar[k].trace_add("write", updateFace)

fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, expand = False, fill = tk.BOTH)

ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.view_init(elev=90,azim=-90)

verts = CC.getVertices()
polyc = ax.plot_trisurf(verts[:,0], verts[:,1], CC.geoTriangles, verts[:,2], shade=True, color='w')
ax.set_box_aspect(np.ptp(verts,axis=0))
ax.set_facecolor('k')
ax.set_axis_off()

root.mainloop()
