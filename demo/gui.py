import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facegen_models import FaceGenSSM

class CharacterCreator (tk.Tk, FaceGenSSM):
    def __init__ (self, lof):
        tk.Tk.__init__(self)
        FaceGenSSM.__init__(self, *lof[0:3])

        self.geoSliders = dict()
        with open(lof[3], 'r') as f: # menu file
            b = f.read()
            rows = b.split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                if len(cells) > 0:
                    slider = CharacterCreator.Slider(*cells)
                    self.geoSliders[cells[0]] = slider

        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Character Creator Demo")

        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.BOTTOM, expand = True, fill = tk.BOTH)

        tabs = dict()
        slivars = dict()
        for k,s in self.geoSliders.items():
            menu = s.menu
            if menu != None:
                if menu not in tabs:
                    tabs[menu] = tk.Frame(notebook)
                    tabs[menu].pack(expand = True, fill = tk.BOTH)
                    notebook.add(tabs[menu], text=menu)
                slivars[k] = tk.IntVar(value=s.guiValue, name=k)
                scale = tk.Scale(tabs[menu], orient=tk.HORIZONTAL, label=s.label, from_=s.guiMin, to=s.guiMax, variable=slivars[k])
                scale.pack(expand = False, fill = 'x')
                slivars[k].trace_add("write", lambda *args: self.updateFace2(args[0]))
        self._tabs = tabs
        self._slivars = slivars

        fig = plt.Figure()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(side=tk.TOP, expand = False, fill = tk.BOTH)
        self._canvas = canvas

        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.view_init(elev=90,azim=-90)

        self.updateSliders()
        verts = self.getVertices()
        self._polyc = ax.plot_trisurf(verts[:,0], verts[:,1], self.geoTriangles, verts[:,2], shade=True, color='w')
        ax.set_box_aspect(np.ptp(verts,axis=0))
        ax.set_facecolor('k')
        ax.set_axis_off()

    def updateSliders (self):
        self._ignoreFlag = True
        for k,s in self.geoSliders.items():
            if s.fgID == 'A':
                value = self.getAge()
            elif s.fgID == 'G':
                value = self.getGender()
            elif s.fgID == 'C':
                value = self.getCaricature()
            else:
                value = self.getSymCtl(s.fgID)
            s.guiValue = round(255 * (value - s.rangeMin) / (s.rangeMax - s.rangeMin))
            if k in self._slivars:
                self._slivars[k].set(s.guiValue)
        self._ignoreFlag = False

    def updateFace1 (self, key): # DeS & DS1
        if self._ignoreFlag:
            return
        s = self.geoSliders[key]
        s.guiValue = self._slivars[key].get()
        value = s.rangeMin + (s.rangeMax - s.rangeMin) * s.guiValue / 255
        if s.fgID == 'A':
            self.setAge(value)
        elif s.fgID == 'G':
            self.setGender(value)
        elif s.fgID == 'C':
            self.setCaricature(value)
        elif s.menu != None:
            self.setSymCtl(s.fgID, value)
        else:
            self.setSymCtl(s.fgID, 0.0)
        self.updateSliders()
        verts = self.getVertices()
        self._polyc.set_verts(verts[self.geoTriangles])
        self._canvas.draw()

    def updateFace2 (self, key): # BB, DS3 & ER
        if self._ignoreFlag:
            return
        self.geoSliders[key].guiValue = self._slivars[key].get()
        self.setZero()
        for k,s in self.geoSliders.items():
            value = s.rangeMin + (s.rangeMax - s.rangeMin) * s.guiValue / 255
            if s.fgID == 'A':
                self.setAge(value)
            elif s.fgID == 'G':
                self.setGender(value)
            elif s.fgID == 'C':
                self.setCaricature(value)
            elif s.menu != None:
                self.setSymCtl(s.fgID, value)
            else:
                self.setSymCtl(s.fgID, 0.0)
        verts = self.getVertices()
        self._polyc.set_verts(verts[self.geoTriangles])
        self._canvas.draw()

    class Slider:
        def __init__ (self, fgID, menu = None, label = None, rangeMin = -10.0, rangeMax = 10.0, guiMin = 0, guiMax = 255):
            try:
                self.fgID = int(fgID)
            except:
                self.fgID = fgID
            self.menu = menu
            self.label = label
            self.rangeMin = float(rangeMin)
            self.rangeMax = float(rangeMax)
            self.guiMin = int(guiMin)
            self.guiMax = int(guiMax)
            self.guiValue = 128

if __name__ == "__main__":
    import sys
    assert(len(sys.argv) == 5)
    root = CharacterCreator(sys.argv[1:]) # CTL, EGM, TRI, CSV
    root.mainloop()
