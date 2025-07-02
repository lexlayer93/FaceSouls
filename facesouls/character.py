from .models import FaceGenerator, FaceGenSAM
from zipfile import is_zipfile, ZipFile

__all__ = [
    "CharacterCreator"
    ]

class CharacterCreator (FaceGenerator):
    def __init__ (self, ctl, menu=None, models=None, *, preset=None, endian=None):
        if not isinstance(ctl, bytes) and is_zipfile(ctl):
            endian, ctl, menu, models = zip_load(ctl)
        super().__init__(ctl, endian=endian)
        self.all_at_once = False
        self.sliders = dict()
        self.values = dict()
        self.tabs = dict()
        self.sequence = list()
        no_limit = (float("-inf"), float("inf"))
        self.shape_sym_range = no_limit
        self.texture_sym_range = no_limit
        self.shape_asym_range = no_limit
        self.texture_asym_range = no_limit

        if models is None:
            self.models = [self.test_model()]
        else:
            self.load_models(*models, endian=endian)
            self.sync_models()

        if preset is not None: # if None, should be zero
            self.load_data(preset, endian=endian)
        self.preset = self.models[0].copy()

        self.load_menu(menu) # sets all_at_once, updates models...

    def load_models (self, *models, endian=None):
        self.models = []
        for sam in models:
            if not isinstance(sam, FaceGenSAM):
                sam = FaceGenSAM(*sam, endian=endian)
            self.fix_model(sam)
            self.models.append(sam)

    def sync_models (self, src=None):
        if src is None: src = self.models[0]
        for sam in self.models:
            if sam == src: continue
            sam.gs_data = src.gs_data
            sam.ts_data = src.ts_data
            sam.ga_data = src.ga_data
            sam.ta_data = src.ta_data

    def load_data (self, fname, *, endian=None):
        try:
            self.models[0].load_data(fname, endian=endian)
        except FileNotFoundError:
            pass
        else:
            self.sync_models()
            self.all_at_once = False
            self.update()

    def save_data (self, fname, *, endian=None):
        self.models[0].save_data(fname, endian=endian)

    def load_values (self, fname):
        csv = csv_load(fname)
        for key, items in csv.items():
            if isinstance(key, int):
                value = int(items[0])
                slider = self.sliders[key]
                self.values[key] = slider.int2float(value)
        if len(csv) != 0:
            self.all_at_once = True
            self.update()

    def save_values (self, fname, *, sort=False):
        out = ""
        if not sort:
            to_save = [slider for key,slider in self.sliders.items() if key in self.sequence]
        else:
            to_save = [self.sliders[key] for tab in self.tabs.values() for key in tab]

        for slider in to_save:
            key = slider.facegen_id
            val = slider.int_value
            lab = slider.label
            tab = slider.tab
            out += f"{key:03d}, {val}, {lab}, {tab};\n"

        with open(fname, 'w') as f:
            f.write(out)

    def export_obj (self, fname):
        out = ""
        o1, o2 = 0, 0
        for i,sam in enumerate(self.models):
            out += f"#{i} model\no part{i}\n"
            out += sam.dump_obj(o1, o2)
            o1 = len(sam.vertices0)
            o2 = len(sam.uv_vertices)
        with open(fname,'w') as f:
            f.write(out)
        return out

    def load_menu (self, src=None):
        self.reset_sliders()
        self.tabs.clear()
        has_sequence = False

        if src is not None:
            menu = csv_load(src)
        else:
            menu = dict()

        for key, items in menu.items():
            if isinstance(key, int):
                tab, label, float_min, float_max, int_min, int_max = items
                slider = self.sliders[key]
                slider.debug_only = False
                slider.label = label
                slider.float_range = (float(float_min), float(float_max))
                slider.int_range = (int(int_min), int(int_max))
                slider.tab = tab
                if tab not in self.tabs:
                    self.tabs[tab] = list()
                self.tabs[tab].append(key)
            elif key=="###":
                self.sequence = [int(k) for k in items]
                has_sequence = True
            elif key=="##1":
                gs_min, gs_max = items
                self.shape_sym_range = (float(gs_min), float(gs_max))
            elif key=="##2":
                ts_min, ts_max = items
                self.texture_sym_range = (float(ts_min), float(ts_max))
            elif key=="##3":
                ga_min, ga_max = items
                self.shape_asym_range = (float(ga_min), float(ga_max))
            elif key=="##4":
                ta_min, ta_max = items
                self.texture_asym_range = (float(ta_min), float(ta_max))

        if len(self.tabs) == 0:
            for key,slider in self.sliders.items():
                tab = slider.tab
                if tab not in self.tabs:
                    self.tabs[tab] = list()
                self.tabs[tab].append(key)

        self.all_at_once = has_sequence
        self.reset_values()
        if not has_sequence:
            self.sequence = [key for tab in self.tabs.values() for key in tab]

    def update (self):
        sam = self.models[0]
        if self.all_at_once:
            sam.copy_data(self.preset)
            for key in self.sequence:
                value = self.values[key]
                self.set_control(key, value, sam)
            self.clip_data(sam)
        else:
            self.clip_data(sam)
            for key in self.values:
                self.values[key] = self.get_control(key, sam)

    def toggle_sequence (self, flag=None):
        if self.all_at_once == flag:
            return
        self.all_at_once = not self.all_at_once
        self.update()

    def set_slider (self, key, value):
        if isinstance(value, int):
            value = self.sliders[key].int2float(value)
        if self.all_at_once:
            self.values[key] = value
            self.update()
        else:
            self.set_control(key, value)
            self.update()

    def clip_data (self, sam=None):
        if sam is None: sam = self.models[0]
        sam.gs_data.clip(*self.shape_sym_range, sam.gs_data)
        sam.ts_data.clip(*self.texture_sym_range, sam.ts_data)
        sam.ga_data.clip(*self.shape_asym_range, sam.ga_data)
        sam.ta_data.clip(*self.texture_asym_range, sam.ta_data)

    def reset_values (self):
        self.values.clear()
        for key,slider in self.sliders.items():
            self.values[key] = sum(slider.float_range)/2
        self.update()

    def reset_sliders (self):
        self.sliders.clear()
        nadd = {100: self.LGS, 200: self.LTS, 300: self.LGA, 400: self.LTA}
        for key in _SETMAP:
            n = nadd.get(key, 1)
            for idx in range(n):
                self.sliders[key+idx] = CharacterSlider(self, key+idx)

    def get_control (self, key, sam=None):
        if sam is None: sam = self.models[0]
        i = key % 100
        k = key - i
        if k != 0:
            return _GETMAP[k](self, i, sam)
        else:
            return _GETMAP[i](self, sam)

    def set_control (self, key, value, sam=None):
        if sam is None: sam = self.models[0]
        i = key % 100
        k = key - i
        if k != 0:
            _SETMAP[k](self, i, value, sam)
        else:
            _SETMAP[i](self, value, sam)


class CharacterSlider:
    def __init__ (self, parent, key):
        self.parent = parent
        self.facegen_id = key
        self.debug_only = True
        if key < 100:
            self.debug_label = _DBLABELS[key]
            self.label = self.debug_label
            k = (key//10)*10
            self.float_range = _DFRANGES[k]
        else:
            k = (key//100)*100
            idx = key % 100
            self.debug_label = getattr(parent, _DBLABELS[k])[idx]
            self.label = self.debug_label
            self.float_range = _DFRANGES[k]
        self.int_range = (0, 255)
        self.tab = _DFTABS[key//100]

    @property
    def available_range (self):
        int_min, int_max = self.int_range
        af_range = self.int2float(int_min), self.int2float(int_max)
        return min(af_range), max(af_range)

    @property
    def value (self):
        return self.parent.values[self.facegen_id]

    @value.setter
    def value (self, x):
        self.parent.values[self.facegen_id] = float(x)

    @property
    def int_value (self):
        return self.float2int(self.value)

    def int2float (self, x):
        float_min, float_max = self.float_range
        return float_min + x*(float_max - float_min)/255

    def float2int (self, x):
        float_min, float_max = self.float_range
        return round(255*(x - float_min)/(float_max - float_min))


def csv_load (src):
    out = dict()

    if isinstance(src, bytes):
        content = src.decode()
    else:
        with open(src, 'r') as f:
            content = f.read()

    rows = list(map(lambda s: s.strip(), content.split(';')))
    for r in rows[:-1]:
        if r.startswith("# "): continue
        cells = tuple(map(lambda s: s.strip(), r.split(',')))
        key = cells[0]
        try:
            out[int(key)] = cells[1:]
        except ValueError:
            out[key] = cells[1:]

    return out


def zip_load (src):
    endian = "big" if src.endswith("be") else "little"
    with ZipFile(src, 'r') as zf:
        lof = sorted(zf.namelist())
        end_ext = "be" if endian=="big" else ""
        csv_list = [f for f in lof if f.endswith(".csv")]
        ctl_list = [f for f in lof if f.endswith(".ctl"+end_ext)]
        tri_list = [f for f in lof if f.endswith(".tri"+end_ext)]
        egm_list = [f for f in lof if f.endswith(".egm"+end_ext)]
        ctl = zf.open(ctl_list[0]).read()
        menu = zf.open(csv_list[0]).read()
        models = list()
        for tri,egm in zip(tri_list, egm_list):
            tri = zf.open(tri).read()
            egm = zf.open(egm).read()
            models.append((tri,egm))
    return endian, ctl, menu, models


_DFTABS = {0: "[Generate]",
           1: "[Shape Symmetry]",
           2: "[Texture Symmetry]",
           3: "[Shape Asymmetry]",
           4: "[Texture Asymmetry]"}

_DFRANGES = {10: (15.0, 60.0),
             20: (-4.0, 4.0),
             30: (0.0, 2.0),
             40: (0.0, 2.0),
             100: (-10.0, 10.0),
             200: (-10.0, 10.0),
             300: (-10.0, 10.0),
             400: (-10.0, 10.0)}

_DBLABELS = {10: "Age",
11: "Age (shape)",
12: "Age (texture)",
13: "Age~",
14: "Age~ (shape)",
15: "Age~ (texture)",
20: "Gender",
21: "Gender (shape)",
22: "Gender (texture)",
23: "Gender~",
24: "Gender~ (shape)",
25: "Gender~ (texture)",
30: "Caricature",
31: "Caricature (shape)",
32: "Caricature (texture)",
33: "Caricature~",
34: "Caricature~ (shape)",
35: "Caricature~ (texture)",
40: "Asymmetry",
41: "Asymmetry (shape)",
42: "Asymmetry (texture)",
100: "lgs_labels",
200: "lts_labels",
300: "lga_labels",
400: "lta_labels"
}

_GETMAP = {10: FaceGenerator.get_age,
11: FaceGenerator.get_shape_age,
12: FaceGenerator.get_texture_age,
13: FaceGenerator.get_age,
14: FaceGenerator.get_shape_age,
15: FaceGenerator.get_texture_age,
20: FaceGenerator.get_gender,
21: FaceGenerator.get_shape_gender,
22: FaceGenerator.get_texture_gender,
23: FaceGenerator.get_gender,
24: FaceGenerator.get_shape_gender,
25: FaceGenerator.get_texture_gender,
30: FaceGenerator.get_caricature_neutral,
31: FaceGenerator.get_shape_caricature_neutral,
32: FaceGenerator.get_texture_caricature_neutral,
33: FaceGenerator.get_caricature,
34: FaceGenerator.get_shape_caricature,
35: FaceGenerator.get_texture_caricature,
40: FaceGenerator.get_asymmetry,
41: FaceGenerator.get_shape_asymmetry,
42: FaceGenerator.get_texture_asymmetry,
100: FaceGenerator.get_shape_sym_control,
200: FaceGenerator.get_texture_sym_control,
300: FaceGenerator.get_shape_asym_control,
400: FaceGenerator.get_texture_asym_control}

_SETMAP = {10: FaceGenerator.set_age_neutral,
11: FaceGenerator.set_shape_age_neutral,
12: FaceGenerator.set_texture_age_neutral,
13: FaceGenerator.set_age,
14: FaceGenerator.set_shape_age,
15: FaceGenerator.set_texture_age,
20: FaceGenerator.set_gender_neutral,
21: FaceGenerator.set_shape_gender_neutral,
22: FaceGenerator.set_texture_gender_neutral,
23: FaceGenerator.set_gender,
24: FaceGenerator.set_shape_gender,
25: FaceGenerator.set_texture_gender,
30: FaceGenerator.set_caricature_neutral,
31: FaceGenerator.set_shape_caricature_neutral,
32: FaceGenerator.set_texture_caricature_neutral,
33: FaceGenerator.set_caricature,
34: FaceGenerator.set_shape_caricature,
35: FaceGenerator.set_texture_caricature,
40: FaceGenerator.set_asymmetry,
41: FaceGenerator.set_shape_asymmetry,
42: FaceGenerator.set_texture_asymmetry,
100: FaceGenerator.set_shape_sym_control,
200: FaceGenerator.set_texture_sym_control,
300: FaceGenerator.set_shape_asym_control,
400: FaceGenerator.set_texture_asym_control}
