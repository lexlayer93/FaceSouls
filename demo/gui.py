import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from facesouls.character import CharacterCreator
from facesouls.tools import facemesh_plot


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
                filetypes=[("Character Creator files","*.zip *.zipbe")],
                defaultextension=".zip")
            self.character_creator = CharacterCreator(path)

        editor = CCMainFrame(self)
        editor.pack(fill=tk.BOTH, expand=True)
        self.mainframe = editor

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
        settings_menu = tk.Menu(self, tearoff=0)
        settings_menu.add_command(label="Reset view",
                                 command=mainframe.canvas.reset)

        _race_menu = tk.Menu(settings_menu, tearoff=0)
        def race_set ():
            r = str(race_set.race.get())
            cc.set_race(r)
            cc.update()
            mainframe.update_all()
        race_set.race = tk.StringVar(value="All")
        for r in ("All", "Afro", "Asia", "Eind", "Euro"):
            _race_menu.add_radiobutton(label=r,
                                        variable=race_set.race,
                                        command=race_set)
        settings_menu.add_cascade(label="Set race", menu=_race_menu)


        def callback ():
            cc.toggle_sequence()
            mainframe.update_all()
            if cc.all_at_once:
                _load_menu.entryconfig(0, state="normal")
                _load_menu.entryconfig(1, state="disabled")
            else:
                _load_menu.entryconfig(0, state="disabled")
                _load_menu.entryconfig(1, state="normal")

        settings_menu.add_checkbutton(label="Interlock suppression",
                                     variable=mainframe._interlock,
                                     command=callback,
                                     )
        _load_menu.entryconfig(1 if cc.all_at_once else 0, state="disabled")

        self.add_cascade(label="File", menu=file_menu)
        self.add_cascade(label="Settings", menu=settings_menu)

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
        path = tk.filedialog.asksaveasfilename(
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
        maxlen = max(len(t) for t in cc.menu)
        for t,tab in cc.menu.items():
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
            svalue = cc.sliders[k].value
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. SHAPE FEATURES:"
        for i in range(cc.LGS):
            k = 100+i
            svalue = cc.sliders[k].value
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        text += "\n\nSYM. TEXTURE FEATURES:"
        for i in range(cc.LTS):
            k = 200+i
            svalue = cc.sliders[k].value
            cvalue = cc.get_control(k)
            label = cc.sliders[k].debug_label
            text += f"\n{i:02d}:[{svalue:{fmt}}]({cvalue:{fmt}})<{label}"

        self.text_left.delete(1.0, tk.END)
        self.text_left.insert(tk.END, text)

        fmt = ".6f"
        face = cc.models[0]
        text = "SYM. SHAPE DATA:"
        for i in range(cc.GS):
            text += f"\n[{i:02d}]: {face.gs_data[i]:{fmt}}"

        text += "\n\nSYM. TEXTURE DATA:"
        for i in range(cc.TS):
            text += f"\n[{i:02d}]: {face.ts_data[i]:{fmt}}"
        self.text_right.delete(1.0, tk.END)
        self.text_right.insert(tk.END, text)


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
