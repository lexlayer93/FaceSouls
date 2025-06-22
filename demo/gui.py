import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot


class CharacterCreatorTK (CharacterCreator):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        main_frame = tk.Frame(parent)
        self.tk_widget = main_frame
        self._tkvars = dict()
        self._canvas = None
        self._polyc = None
        self._debug_text = list()

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        style = ttk.Style()
        style.configure("TNotebook", tabposition="wn")
        style.configure("TNotebook.Tab", font="TkFixedFont")

        notebook = ttk.Notebook(main_frame)
        notebook.config(width=300)
        notebook.grid(row=0, column=0, sticky="nsew")

        maxlen = max(len(t) for t in self.tabs)
        self._callback = lambda key,*args, cc=self: cc.set_slider(key)
        for t,tab in self.tabs.items():
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t.center(maxlen))
            for key in tab:
                slider = self.sliders[key]
                var = tk.IntVar(value=slider.int_value, name=str(key))
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=var)
                scale.pack(expand=False, fill='x')
                var.trace_add("write", self._callback)
                self._tkvars[key] = var

        fig = plt.figure(figsize=plt.figaspect(1), facecolor='k')
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(bg="black", highlightthickness=0, bd=0)
        canvas_widget.grid(row=0, column=1, sticky="nsew")
        self._canvas = canvas
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = self.models[0].vertices
        triangles = self.models[0].triangles_only
        self._polyc = facemesh_plot((verts, triangles), ax, persp="persp")
        self._ignore = False

        dbframe1 = tk.Frame(main_frame, relief="sunken", borderwidth=2)
        dbframe1.grid(row=1, column=0, sticky="nsew")
        scrollbar1 = tk.Scrollbar(dbframe1)
        scrollbar1.pack(side=tk.LEFT, fill=tk.Y)
        text1 = tk.Text(dbframe1, wrap=tk.WORD)
        text1.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        text1.config(yscrollcommand=scrollbar1.set)
        scrollbar1.config(command=text1.yview)

        dbframe2 = tk.Frame(main_frame, relief="sunken", borderwidth=2)
        dbframe2.grid(row=1, column=1, sticky="nsew")
        scrollbar2 = tk.Scrollbar(dbframe2)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        text2 = tk.Text(dbframe2, wrap=tk.WORD)
        text2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text2.config(yscrollcommand=scrollbar2.set)
        scrollbar2.config(command=text2.yview)
        self._debug_text = [text1, text2]
        self.update_debug()

    def set_slider (self, key, value=None):
        if self._ignore:
            return
        key = int(key)
        value = value or self._tkvars[key].get()
        super().set_slider(key, value)
        self.update_all()

    def toggle_sequence (self, *args, **kwargs):
        super().toggle_sequence(*args, **kwargs)
        self.update_all()

    def update_all (self):
        self.update_canvas()
        self.update_tkvars()
        self.update_debug()

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

    def update_debug (self, fmt=".3f"):
        face = self.models[0]
        text = "FEATURES:"
        keys = (10, 20, 30)
        for k in keys:
            svalue = self.values[k]
            cvalue = self.get_control(k)
            label = self.sliders[k].debug_label
            text += f"\n[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. SHAPE CONTROLS:"
        for i in range(self.LGS):
            k = 100+i
            svalue = self.values[k]
            cvalue = self.get_control(k)
            label = self.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. TEXTURE CONTROLS:"
        for i in range(self.LTS):
            k = 200+i
            svalue = self.values[k]
            cvalue = self.get_control(k)
            label = self.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        self._debug_text[0].delete(1.0, tk.END)
        self._debug_text[0].insert(tk.END, text)

        text = "SYM. SHAPE DATA:"
        for i in range(self.GS):
            text += f"\n[{i:02d}]: {face.gs_data[i]:{fmt}}"

        text += "\n\nSYM. TEXTURE DATA:"
        for i in range(self.TS):
            text += f"\n[{i:02d}]: {face.ts_data[i]:{fmt}}"
        self._debug_text[1].delete(1.0, tk.END)
        self._debug_text[1].insert(tk.END, text)


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
        elif extension == "fg":
            self.load_data(path)
        self.update_all()

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
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.title("Character Creator Demo")

    ctl, csv, tri, egm = sys.argv[1:5] # si.ctl, ds3.csv, FG_A_0100.tri, FG_A_0100.egm
    face = (tri, egm)
    cc = CharacterCreatorTK(root, ctl, csv, [face])
    cc.tk_widget.pack(expand=True, fill=tk.BOTH)

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
