import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot


class CharacterCreatorTk (tk.Tk):
    def __init__ (self, char_creator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Character Creator Demo")

        cc = char_creator
        self._cc = cc

        editor = CCEditorFrame(self)
        editor.pack(fill=tk.BOTH, expand=True)
        self._editor = editor

        menubar = CCMenu(self)
        self.config(menu=menubar)
        self._menubar = menubar


class CCMenu (tk.Menu):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        editor = parent._editor

        file_menu = tk.Menu(self, tearoff=0)
        _load_menu = tk.Menu(file_menu, tearoff=0)
        _load_menu.add_command(label="Sliders values", command=self.ask_load_values)
        _load_menu.add_command(label="FaceGen FG", command=self.ask_load_data)
        _save_menu = tk.Menu(file_menu, tearoff=0)
        _save_menu.add_command(label="Sliders values", command=self.ask_save_values)
        _save_menu.add_command(label="FaceGen FG", command=self.ask_save_data)
        _export_menu = tk.Menu(file_menu, tearoff=0)
        _export_menu.add_command(label="Wavefront OBJ", command=self.ask_export_obj)
        file_menu.add_cascade(label="Load...", menu=_load_menu)
        file_menu.add_cascade(label="Save...", menu=_save_menu)
        file_menu.add_cascade(label="Export...", menu=_export_menu)

        options_menu = tk.Menu(self, tearoff=0)
        options_menu.add_checkbutton(label="Slider interlock",
                                     variable=editor._interlock,
                                     command=editor.interlock_callback,
                                     )

        self.add_cascade(label="File", menu=file_menu)
        self.add_cascade(label="Options", menu=options_menu)


    def ask_load_values (self):
        path = tk.filedialog.askopenfilename(
            title = "Load...",
            filetypes = [("Sliders values", "*.csv")],
            defaultextension=".csv"
            )
        self.master._cc.load_values(path)
        self.master._editor.update_all()

    def ask_save_values (self):
        path = tk.filedialog.asksaveasfilename(
            title = "Save...",
            filetypes = [("Sliders values", "*.csv")],
            defaultextension=".csv"
            )
        self.master._cc.save_values(path)

    def ask_load_data (self):
        path = tk.filedialog.askopenfilename(
            title = "Load...",
            filetypes = [("FaceGen FG", "*.fg")],
            defaultextension=".fg"
            )
        self.master._cc.load_data(path)
        self.master._editor.update_all()

    def ask_save_data (self):
        path = tk.filedialog.askopenfilename(
            title = "Save...",
            filetypes = [("FaceGen FG", "*.fg")],
            defaultextension=".fg"
            )
        self.master._cc.save_data(path)

    def ask_export_obj (self):
        path = tk.filedialog.askopenfilename(
            title = "Export...",
            filetypes = [("Wavefront OBJ", "*.obj")],
            defaultextension=".obj"
            )
        self.master._cc.models[0].export_obj(path)


class CCEditorFrame (tk.Frame):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        cc = parent._cc
        self._cc = cc
        self._tkvars = dict()
        self._canvas = None
        self._polyc = None
        self._interlock = tk.BooleanVar(value=not cc.all_at_once)
        self._debug_text = list()

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        style = ttk.Style()
        style.configure("TNotebook", tabposition="wn")
        style.configure("TNotebook.Tab", font="TkFixedFont")

        notebook = ttk.Notebook(self)
        notebook.config(width=300)
        notebook.grid(row=0, column=0, sticky="nsew")

        maxlen = max(len(t) for t in cc.tabs)
        for t,tab in cc.tabs.items():
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t.center(maxlen))
            for key in tab:
                slider = cc.sliders[key]
                int_var = tk.IntVar(value=slider.int_value, name=str(key))
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=int_var)
                scale.pack(expand=False, fill='x')
                int_var.trace_add("write", self.slider_callback)
                self._tkvars[key] = int_var

        fig = plt.figure(figsize=plt.figaspect(1), facecolor='k')
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(bg="black", highlightthickness=0, bd=0)
        canvas_widget.grid(row=0, column=1, sticky="nsew")
        self._canvas = canvas
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = cc.models[0].vertices
        triangles = cc.models[0].triangles_only
        self._polyc = facemesh_plot((verts, triangles), ax, persp="persp")
        self._ignore = False

        dbframe1 = tk.Frame(self)
        dbframe1.grid(row=1, column=0, sticky="nsew")
        scrollbar1 = tk.Scrollbar(dbframe1)
        scrollbar1.pack(side=tk.LEFT, fill=tk.Y)
        text1 = tk.Text(dbframe1, wrap=tk.WORD)
        text1.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        text1.config(yscrollcommand=scrollbar1.set)
        scrollbar1.config(command=text1.yview)

        dbframe2 = tk.Frame(self)
        dbframe2.grid(row=1, column=1, sticky="nsew")
        scrollbar2 = tk.Scrollbar(dbframe2)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        text2 = tk.Text(dbframe2, wrap=tk.WORD)
        text2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text2.config(yscrollcommand=scrollbar2.set)
        scrollbar2.config(command=text2.yview)
        self._debug_text = [text1, text2]
        self.update_debug()

    def slider_callback (self, key, *args):
        if self._ignore:
            return
        key = int(key)
        value = self._tkvars[key].get()
        cc = self.master._cc
        cc.set_slider(key, value)
        self.update_all()

    def interlock_callback (self, flag=None):
        cc = self.master._cc
        cc.toggle_sequence(flag)
        self.update_all()

    def update_all (self):
        self.update_canvas()
        self.update_debug()
        self.update_tkvars()
        cc = self.master._cc
        self._interlock.set(not cc.all_at_once)

    def update_canvas (self):
        cc = self.master._cc
        verts = cc.models[0].vertices
        triangles = cc.models[0].triangles_only
        self._polyc.set_verts(verts[triangles])
        self._canvas.draw()

    def update_tkvars (self):
        self._ignore = True
        cc = self.master._cc
        for key, int_var in self._tkvars.items():
            value = cc.sliders[key].int_value
            int_var.set(value)
        self._ignore = False

    def update_debug (self, fmt=".3f"):
        cc = self.master._cc
        text = "FEATURES:"
        for k in (10, 20, 30):
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. SHAPE CONTROLS:"
        for i in range(cc.LGS):
            k = 100+i
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. TEXTURE CONTROLS:"
        for i in range(cc.LTS):
            k = 200+i
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        self._debug_text[0].delete(1.0, tk.END)
        self._debug_text[0].insert(tk.END, text)

        face = cc.models[0]
        text = "SYM. SHAPE DATA:"
        for i in range(cc.GS):
            text += f"\n{i:02d}:({face.gs_data[i]:{fmt}})"

        text += "\n\nSYM. TEXTURE DATA:"
        for i in range(cc.TS):
            text += f"\n{i:02d}:({face.ts_data[i]:{fmt}})"
        self._debug_text[1].delete(1.0, tk.END)
        self._debug_text[1].insert(tk.END, text)

if __name__ == "__main__":
    import sys
    assert len(sys.argv) >= 5

    ctl, csv, tri, egm = sys.argv[1:5] # si.ctl, ds3.csv, FG_A_0100.tri, FG_A_0100.egm
    face = (tri, egm)
    cc = CharacterCreator(ctl, csv, [face])

    CharacterCreatorTk(cc).mainloop()
