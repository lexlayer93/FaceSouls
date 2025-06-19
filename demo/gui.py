import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot


class CharacterCreatorTK (CharacterCreator, tk.Tk):
    def __init__ (self, ctl, menu, face, preset=None, endian="little"):
        tk.Tk.__init__(self)
        self._tkvars = dict()
        self._canvas = None
        self._polyc = None

        CharacterCreator.__init__(self, ctl, menu, [face], preset, endian)

        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Character Creator Demo")

        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.ask_open)
        file_menu.add_command(label="Save as...", command=self.ask_save)
        file_menu.add_command(label="Export as...", command=self.ask_export)

        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(label="Toggle mode", command=self.toggle_sequence)

        self.config(menu=menubar)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Options", menu=options_menu)

        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.BOTTOM, fill = tk.BOTH, expand=True)

        for t in self.tabs:
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t)
            for key in self.tabs[t]:
                slider = self.sliders[key]
                var = tk.IntVar(value=slider.int_value, name=str(key))
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=var)
                scale.pack(expand=False, fill='x')
                var.trace_add("write", lambda key,*args, cc=self: cc.set_slider(key))
                self._tkvars[key] = var

        fig = plt.Figure()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self._canvas = canvas
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = self.models[0].vertices
        triangles = self.models[0].triangles_only
        self._polyc = facemesh_plot((verts, triangles), ax)
        self._ignore = False

    def update_canvas (self):
        verts = self.models[0].vertices
        triangles = self.models[0].triangles_only
        self._polyc.set_verts(verts[triangles])
        self._canvas.draw()

    def update_tkvars (self):
        self._ignore = True
        for key, int_var in self._tkvars.items():
            value = self.sliders[key].int_value
            int_var.set(value)
        self._ignore = False

    def set_slider (self, key, value=None):
        if self._ignore:
            return
        key = int(key)
        value = value or self._tkvars[key].get()
        super().set_slider(key, value)
        self.update_canvas()

    def update_values (self):
        super().update_values()
        self.update_tkvars()

    def toggle_sequence (self, *args, **kwargs):
        super().toggle_sequence(*args, **kwargs)
        self.update_canvas()

    def ask_open (self):
        path = tk.filedialog.askopenfilename(
            title = "Open...",
            filetypes = [("Sliders values", "*.csv"),
                         ("FaceGen FG", "*.fg")],
            defaultextension=".csv"
            )
        extension = path.split('.')[-1]
        if extension == "csv":
            self.load_values(path)
            self.update_tkvars()
        elif extension == "fg":
            self.load_data(path)
        self.update_canvas()

    def ask_save (self):
        path = tk.filedialog.asksaveasfilename(
            title = "Save...",
            filetypes = [("Sliders values", "*.csv"),
                         ("FaceGen FG", "*.fg"),
                         ("Wavefront OBJ", "*.obj")],
            defaultextension=".csv"
            )
        extension = path.split('.')[-1]
        if extension == "csv":
            self.save_values(path, sort=False)
        elif extension == "fg":
            self.save_data(path)
        elif extension == "obj":
            self.models[0].export_as_obj(path)

    def ask_export (self):
        path = tk.filedialog.asksaveasfilename(
            title = "Export...",
            filetypes = [("Wavefront OBJ", "*.obj")],
            defaultextension=".obj"
            )
        extension = path.split('.')[-1]
        if extension == "obj":
            self.models[0].export_as_obj(path)

if __name__ == "__main__":
    import sys
    assert len(sys.argv) >= 5
    ctl, csv, tri, egm = sys.argv[1:5] # si.ctl, ds3.csv, FG_A_0100.tri, FG_A_0100.egm
    preset = sys.argv[5] if len(sys.argv) >= 6 else None
    face = (tri, egm)
    root = CharacterCreatorTK(ctl, csv, face, preset)
    root.mainloop()
