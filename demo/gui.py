import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot

class CharacterCreatorTK (CharacterCreator, tk.Tk):
    def __init__ (self, ctl, menu, face, data=None, endian="little"):
        tk.Tk.__init__(self)
        self._tkvars = dict()
        self._canvas = None
        self._polyc = None
        CharacterCreator.__init__(self, ctl, menu, [face], data, endian)

        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Character Creator Demo")

        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.BOTTOM, fill = tk.BOTH, expand=True)
        for t in self.tabs:
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t)
            for fg_id in self.tabs[t]:
                slider = self.sliders[fg_id]
                var = tk.IntVar(value=slider.int_value, name=fg_id)
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=var)
                scale.pack(expand=False, fill='x')
                var.trace_add("write", lambda fg_id,*args, self=self: self.set_slider(fg_id))
                self._tkvars[fg_id] = var

        fig = plt.Figure()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self._canvas = canvas
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = self.models[0].vertices
        triangles = self.models[0].triangles
        self._polyc = facemesh_plot((verts, triangles), ax)
        self._ignore = False

    def set_slider (self, fg_id, value=None):
        if self._ignore:
            return
        int_var = self._tkvars[fg_id]
        value = value or int_var.get()
        super().set_slider(fg_id, value)
        verts = self.models[0].vertices
        triangles = self.models[0].triangles
        self._polyc.set_verts(verts[triangles])
        self._canvas.draw()

    def update_sliders (self, src=None):
        super().update_sliders(src)
        self._ignore = True
        for fg_id, int_var in self._tkvars.items():
            value = self.sliders[fg_id].int_value
            int_var.set(value)
        self._ignore = False

if __name__ == "__main__":
    import sys
    from facesouls.models import FaceGenSAM
    assert len(sys.argv) == 5
    ctl, csv, tri, egm = sys.argv[1:] # si.ctl, ds3.csv, FG_A_0100.tri, FG_A_0100.egm
    face = FaceGenSAM(tri, egm)
    root = CharacterCreatorTK(ctl, csv, face)
    root.mainloop()
