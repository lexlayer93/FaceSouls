import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot, cc_target_shape
from zipfile import ZipFile


class CharacterCreatorTk (tk.Tk):
    def __init__ (self, cc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.character_creator = cc
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.title("Character Creator Demo")
        self.withdraw()

    def mainloop (self):
        if self.character_creator is None:
            path = tk.filedialog.askopenfilename(parent=self,
                title="Open...",
                filetypes=[("Character Creator files","*.zip")],
                defaultextension=".zip")
            with ZipFile(path, 'r') as zf:
                lof = sorted(zf.namelist())
                ctl_list = [f for f in lof if f.endswith(".ctl")]
                csv_list = [f for f in lof if f.endswith(".csv")]
                tri_list = [f for f in lof if f.endswith(".tri")]
                ctl = zf.open(ctl_list[0]).read()
                csv = zf.open(csv_list[0]).read()
                models = list()
                for tri in tri_list:
                    basename = tri.split('.',1)[0]
                    egm = basename + ".egm"
                    tri = zf.open(tri).read()
                    egm = zf.open(egm).read()
                    models.append((tri,egm))
            self.character_creator = CharacterCreator(ctl, csv, models)

        editor = CCMainFrame(self)
        editor.pack(fill=tk.BOTH, expand=True)
        self.mainframe = editor

        repfg = CCReplicateFG(self)
        repfg.withdraw()
        self.top_repfg = repfg

        menubar = CCMainMenu(self)
        self.config(menu=menubar)
        self.mainmenu = menubar

        self.deiconify()
        super().mainloop()


class CCMainMenu (tk.Menu):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        cc = parent.character_creator

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


        mainframe = parent.mainframe
        options_menu = tk.Menu(self, tearoff=0)
        options_menu.add_command(label="Reset view",
                                 command=mainframe.canvas.reset)

        def callback ():
            cc.toggle_sequence()
            mainframe.update_all()
            if cc.all_at_once:
                _load_menu.entryconfig(0, state="normal")
                _load_menu.entryconfig(1, state="disabled")
            else:
                _load_menu.entryconfig(0, state="disabled")
                _load_menu.entryconfig(1, state="normal")

        options_menu.add_checkbutton(label="Interlock suppression",
                                     variable=mainframe._interlock,
                                     command=callback,
                                     )
        _load_menu.entryconfig(1 if cc.all_at_once else 0, state="disabled")

        tools_menu = tk.Menu(self, tearoff=0)
        tools_menu.add_command(label="FG Replication",
                               command=parent.top_repfg.deiconify)

        self.add_cascade(label="File", menu=file_menu)
        self.add_cascade(label="Options", menu=options_menu)
        self.add_cascade(label="Tools", menu=tools_menu)

    def ask_load_values (self):
        cc = self.master.character_creator
        mainframe = self.master.mainframe
        path = tk.filedialog.askopenfilename(
            title = "Load...",
            filetypes = [("Sliders values", "*.csv")],
            defaultextension=".csv"
            )
        cc.load_values(path)
        mainframe.update_all()

    def ask_save_values (self):
        cc = self.master.character_creator
        path = tk.filedialog.asksaveasfilename(
            title = "Save...",
            filetypes = [("Sliders values", "*.csv")],
            defaultextension=".csv"
            )
        cc.save_values(path)

    def ask_load_data (self):
        cc = self.master.character_creator
        mainframe = self.master.mainframe
        path = tk.filedialog.askopenfilename(
            title = "Load...",
            filetypes = [("FaceGen FG", "*.fg")],
            defaultextension=".fg"
            )
        cc.load_data(path)
        mainframe.update_all()

    def ask_save_data (self):
        cc = self.master.character_creator
        mainframe = self.master.mainframe
        path = tk.filedialog.askopenfilename(
            title = "Save...",
            filetypes = [("FaceGen FG", "*.fg")],
            defaultextension=".fg"
            )
        cc.save_data(path)

    def ask_export_obj (self):
        cc = self.master.character_creator
        path = tk.filedialog.asksaveasfilename(
            title = "Export...",
            filetypes = [("Wavefront OBJ", "*.obj")],
            defaultextension=".obj"
            )
        cc.export_obj(path)


class CCMainFrame (tk.Frame):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        cc = parent.character_creator

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        style = ttk.Style()
        style.configure("TNotebook", tabposition="wn")
        style.configure("TNotebook.Tab", font="TkFixedFont")

        # Notebook widget
        notebook = ttk.Notebook(self, width=300)
        #notebook.config(width=300)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._slivars = dict()
        maxlen = max(len(t) for t in cc.tabs)
        for t,tab in cc.tabs.items():
            frame = tk.Frame(notebook)
            frame.pack(expand=True, fill=tk.BOTH)
            notebook.add(frame, text=t.center(maxlen))
            for key in tab:
                slider = cc.sliders[key]
                int_var = tk.IntVar(value=slider.int_value, name=str(key))
                self._slivars[key] = int_var
                def callback (value, k=key):
                    cc.set_slider(k, int(value))
                    self.update_all()
                scale = tk.Scale(frame, orient=tk.HORIZONTAL, label=slider.label,
                                 from_=slider.int_range[0], to=slider.int_range[1],
                                 variable=int_var,
                                 command=callback)
                scale.pack(expand=False, fill='x')

        self._interlock = tk.BooleanVar(value=cc.all_at_once)
        self.notebook = notebook

        # Canvas widget
        canvas = CCCanvas(self, *cc.models)
        canvas.widget.grid(row=0, column=1, sticky="nsew")
        self.canvas = canvas

        # Left debug text widget
        frame = tk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        text = tk.Text(frame, wrap=tk.WORD)
        text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)
        self.text_left = text

        # Right debug text widget
        frame = tk.Frame(self)
        frame.grid(row=1, column=1, sticky="nsew")
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        text = tk.Text(frame, wrap=tk.WORD)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text.yview)
        self.text_right = text

        self.update_debug()

    def update_all (self):
        cc = self.master.character_creator
        self.canvas.update()
        self.update_debug()
        self.update_sliders()
        self._interlock.set(cc.all_at_once)

    def update_sliders (self):
        cc = self.master.character_creator
        for key, int_var in self._slivars.items():
            value = cc.sliders[key].int_value
            int_var.set(value)

    def update_debug (self):
        cc = self.master.character_creator
        fmt = ".3f"
        text = "GENERAL FEATURES:"
        for k in (10, 20, 30):
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. SHAPE FEATURES:"
        for i in range(cc.LGS):
            k = 100+i
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. TEXTURE FEATURES:"
        for i in range(cc.LTS):
            k = 200+i
            svalue = cc.values[k]
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        self.text_left.delete(1.0, tk.END)
        self.text_left.insert(tk.END, text)

        face = cc.models[0]
        text = "SYM. SHAPE DATA:"
        for i in range(cc.GS):
            text += f"\n{i:02d}:({face.gs_data[i]:{fmt}})"

        text += "\n\nSYM. TEXTURE DATA:"
        for i in range(cc.TS):
            text += f"\n{i:02d}:({face.ts_data[i]:{fmt}})"
        self.text_right.delete(1.0, tk.END)
        self.text_right.insert(tk.END, text)


class CCReplicateFG (tk.Toplevel):
    def __init__ (self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("FG Replication")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        cc = parent.character_creator
        # Target
        self.target = cc.models[0].copy()
        canvas = CCCanvas(self)
        label = tk.Label(self, text="Target", font=("TkDefaultFont",16,"bold"))
        button = tk.Button(self, text="Load FG", command=self.ask_load_target)

        label.grid(row=0, column=0, columnspan=3, sticky="ew")
        canvas.widget.grid(row=1, column=0, columnspan=3, sticky="nsew",padx=1)
        button.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.canvas_target = canvas

        # Replica
        self.replica = cc.models[0].copy()
        canvas = CCCanvas(self)
        label = tk.Label(self, text="Replica", font=("TkDefaultFont",16,"bold"))
        button = tk.Button(self, text="Save CSV", command=self.ask_save_replica)

        label.grid(row=0, column=3, columnspan=3, sticky="ew")
        canvas.widget.grid(row=1, column=3, columnspan=3, sticky="nsew",padx=1)
        button.grid(row=2, column=3, columnspan=3, sticky="ew")
        self.canvas_replica = canvas

        # Options
        combo = ttk.Combobox(self,
                             values=("Fit shape data",
                                     "Fit shape features",
                                     "Fit shape mesh"),
                             state="readonly")
        combo.current(0)
        self.combobox = combo
        label1 = tk.Label(self, text="Max. Iterations:")
        scale = tk.Scale(self,
                         orient=tk.HORIZONTAL,
                         showvalue=False,
                         from_=100, to=1000,
                         resolution=100,
                         command=lambda x: label2.config(text=x.ljust(4)))
        self.iterations = scale
        label2 = tk.Label(self, text="100 ", font="TkFixedFont")
        button = tk.Button(self, text="Confirm", command=self.find_replica,
                           state="disabled")
        self.confirm = button
        combo.grid(row=3,column=0,sticky="w")
        label1.grid(row=3,column=1,sticky="w")
        scale.grid(row=3, column=2, columnspan=2, sticky="ew")
        label2.grid(row=3,column=4,sticky="e")
        button.grid(row=3,column=5,sticky="e")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)

    def find_replica (self):
        cc = self.master.character_creator
        current_mode = self.combobox.current()
        current_iter = int(self.iterations.get())
        solution, data, info = cc_target_shape(cc,
                                               self.target.gs_data,
                                               mode=current_mode,
                                               options={"maxiter":current_iter})
        self.solution = solution
        self.replica.gs_data = data
        self.canvas_replica.replot(self.replica)
        if info.success:
            text = info.message + f"\nIterations: {info.nit}"
            tk.messagebox.showinfo("SUCCESS!", text, parent=self)
        else:
            text = info.message
            tk.messagebox.showwarning("FAILURE!", text, parent=self)

    def ask_load_target (self):
        path = tk.filedialog.askopenfilename(parent=self,
            title = "Load target...",
            filetypes = [("FaceGen FG", "*.fg")],
            defaultextension=".fg"
            )
        self.target.load_data(path)
        self.canvas_target.replot(self.target)
        self.confirm.config(state="normal")

    def ask_save_replica (self):
        path = tk.filedialog.asksaveasfilename(parent=self,
            title = "Save replica...",
            filetypes = [("Sliders values", "*.csv")],
            defaultextension=".csv"
            )
        cc = self.master.character_creator
        out = ""
        for key,value in self.solution.items():
            slider = cc.sliders[key]
            value = slider.float2int(value)
            label = slider.label
            tab = slider.tab
            out += f"{key:03d}, {value}, {label}, {tab};\n"
        with open(path, 'w') as f:
            f.write(out)


class CCCanvas (FigureCanvasTkAgg):
    def __init__ (self, parent, *models, width=480, **kwargs):
        kwargs.setdefault("dpi", 100)
        kwargs.setdefault("facecolor", "black")
        width = width/kwargs["dpi"]
        self.figure = plt.figure(figsize=(width,width), **kwargs)
        super().__init__(self.figure, parent)
        self.collection = list()
        self.replot(*models)
        widget = self.get_tk_widget()
        widget.config(bg=kwargs["facecolor"])
        self.widget = widget

    def replot (self, *models):
        if len(models) == 0:
            return
        fig = self.figure
        for ax in fig.axes:
            ax.remove()
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        self.collection.clear()
        for sam in models:
            verts = sam.vertices
            triangles = sam.triangles_only
            poly3d = facemesh_plot((verts, triangles), ax, persp="persp")
            self.collection.append((sam,poly3d))
        self.draw()

    def update (self):
        for sam,poly3d in self.collection:
            verts = sam.vertices
            triangles = sam.triangles_only
            poly3d.set_verts(verts[triangles])
        self.draw()

    def reset (self):
        for ax in self.figure.axes:
            ax.view_init(vertical_axis='y', elev=0, azim=0, roll=0)
        self.draw()


if __name__ == "__main__":
    CharacterCreatorTk().mainloop()
