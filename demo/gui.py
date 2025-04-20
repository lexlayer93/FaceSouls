import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls import *
import sys

def updateFace (key = None, *args):
    ccBB.update2(key, tkvars[key].get())
    verts = ccBB.getVertices()

    polyc.set_verts(verts[ccBB.vtxTriangles])
    canvas.draw()

assert(len(sys.argv)>4)
ctlfname = sys.argv[1]
egmfname = sys.argv[2]
trifname = sys.argv[3]
menufname = sys.argv[4]
ccBB = CharacterCreator(ctlfname, egmfname, trifname, menufname)

root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", root.quit)
notebook = ttk.Notebook(root)
notebook.pack(side=tk.BOTTOM, expand = 1, fill = tk.BOTH)


tabs = dict()
tkvars = dict()

for k,s in ccBB.geoSliders.items():
    menu = s.menu
    if menu != None and menu not in tabs:
        tabs[menu] = tk.Frame(notebook)
        tabs[menu].pack(expand = True, fill = tk.BOTH)
        notebook.add(tabs[menu], text=menu)

    if menu != None:
        tkvars[k] = tk.IntVar(value=s.guiValue, name=k)
        sca = tk.Scale(tabs[menu], orient=tk.HORIZONTAL, label=s.label, from_=s.guiMin, to=s.guiMax, variable=tkvars[k])
        sca.pack(expand = False, fill = 'x')
        tkvars[k].trace_add("write", updateFace)

if False:
    with open('bb.csv') as f:
        s = f.read()
        rows = s.split(';')
        for r in rows:
            cells = list(map(lambda s: s.strip(), r.split(',')))
            if len(cells) < 7:
                continue
            key = cells[1]
            if key not in tabs:
                tabs[key] = tk.Frame(notebook)
            idx = cells[0]
            slis[idx] = (tk.IntVar(value=128, name=idx), cells[1], cells[2], float(cells[3]), float(cells[4]), int(cells[5]), int(cells[6]))

    for k,v in tabs.items():
        v.pack(expand = True, fill = tk.BOTH)
        notebook.add(v, text=k)

    for k,v in slis.items():
        aux = tk.Scale(tabs[v[1]], orient=tk.HORIZONTAL, label=v[2], from_=v[5], to=v[6], \
            variable = v[0])
        aux.pack(expand = False, fill = "x")
        v[0].trace_add('write',updateFace)

fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, expand = False, fill = tk.BOTH)

ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.view_init(elev=90,azim=-90)

verts0 = ccBB.getVertices()
polyc = ax.plot_trisurf(verts0[:,0], verts0[:,1], ccBB.vtxTriangles, verts0[:,2], shade=True, color='w')
ax.set_box_aspect(np.ptp(verts0,axis=0))
ax.set_facecolor('k')
ax.set_axis_off()

root.mainloop()
