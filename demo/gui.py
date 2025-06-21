import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot


class CharacterCreatorTK (tk.Frame):
    def __init__ (self, character_creator, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        cc = character_creator
        self._cc = cc
        self._tkvars = dict()
        self._canvas = None
        self._polyc = None

        style = ttk.Style()
        style.configure("TNotebook", tabposition="wn")
        style.configure("TNotebook.Tab", font="TkFixedFont")

        notebook = ttk.Notebook(self)
        notebook.config(width=300)
        notebook.pack(side=tk.LEFT, fill = tk.BOTH, expand=True)

        maxlen = max(len(t) for t in cc.tabs)
        for t,tab in cc.tabs.items():
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t.center(maxlen))
            for key in tab:
                slider = cc.sliders[key]
                var = tk.IntVar(value=slider.int_value, name=str(key))
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=var)
                scale.pack(expand=False, fill='x')
                var.trace_add("write", lambda key,*args, cc=self: self.set_slider(key))
                self._tkvars[key] = var

        fig = plt.figure(figsize=plt.figaspect(1), facecolor='k')
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(bg="black", highlightthickness=0, bd=0)
        canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._canvas = canvas
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = cc.models[0].vertices
        triangles = cc.models[0].triangles_only
        self._polyc = facemesh_plot((verts, triangles), ax, persp="persp")
        self._ignore = False

    def update_canvas (self):
        verts = self._cc.models[0].vertices
        triangles = self._cc.models[0].triangles_only
        self._polyc.set_verts(verts[triangles])
        self._canvas.draw()

    def update_tkvars (self):
        self._ignore = True
        for key, int_var in self._tkvars.items():
            value = self._cc.sliders[key].int_value
            int_var.set(value)
        self._ignore = False

    def set_slider (self, key, value=None):
        if self._ignore:
            return
        key = int(key)
        value = value or self._tkvars[key].get()
        self._cc.set_slider(key, value)
        self.update_canvas()
        if not self._cc.all_at_once:
            self.update_tkvars()

    def toggle_sequence (self, *args, **kwargs):
        self._cc.toggle_sequence(*args, **kwargs)
        self.update_canvas()
        self.update_tkvars()

    def ask_open (self):
        path = tk.filedialog.askopenfilename(
            title = "Open...",
            filetypes = [("Sliders values", "*.csv"),
                         ("FaceGen FG", "*.fg")],
            defaultextension=".csv"
            )
        extension = path.split('.')[-1]
        if extension == "csv":
            self._cc.load_values(path)
        elif extension == "fg":
            self._cc.load_data(path)
        self.update_canvas()
        self.update_tkvars()

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
            self._cc.save_values(path, sort=False)
        elif extension == "fg":
            self._cc.save_data(path)
        elif extension == "obj":
            self._cc.models[0].export_as_obj(path)

    def ask_export (self):
        path = tk.filedialog.asksaveasfilename(
            title = "Export...",
            filetypes = [("Wavefront OBJ", "*.obj")],
            defaultextension=".obj"
            )
        extension = path.split('.')[-1]
        if extension == "obj":
            self._cc.models[0].export_as_obj(path)

if __name__ == "__main__":
    import sys
    assert len(sys.argv) >= 5
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.title("Character Creator Demo")

    ctl, csv, tri, egm = sys.argv[1:5] # si.ctl, ds3.csv, FG_A_0100.tri, FG_A_0100.egm
    face = (tri, egm)
    cc = CharacterCreator(ctl, csv, [face])
    cc = CharacterCreatorTK(cc, parent=root)
    cc.pack(expand=True, fill=tk.BOTH)

    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open...", command=cc.ask_open)
    file_menu.add_command(label="Save as...", command=cc.ask_save)
    file_menu.add_command(label="Export as...", command=cc.ask_export)

    options_menu = tk.Menu(menubar, tearoff=0)
    options_menu.add_command(label="Toggle mode", command=cc.toggle_sequence)

    root.config(menu=menubar)
    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label="Options", menu=options_menu)

    root.mainloop()
