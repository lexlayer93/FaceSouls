from .models import FaceGenerator, FaceGenSAM
from zipfile import ZipFile

__all__ = [
    "CharacterCreator"
    ]

class CharacterCreator (FaceGenerator):
    def __init__ (self, ctl, menu=None, models=None, *, preset=None, endian=None):
        super().__init__(ctl, endian=endian)
        self.all_at_once = False
        self.sliders = dict()
        self.menu = dict()
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

        self.load_menu(menu) # sets all_at_once, updates models...

    @classmethod
    def fromzip (cls, src):
        with ZipFile(src, 'r') as zf:
            lof = sorted(zf.namelist())
            lof = [tuple(f.rsplit('.',1)) for f in lof]
            csv_list = ['.'.join(f) for f in lof if f[-1].startswith("csv")]
            ctl_list = ['.'.join(f) for f in lof if f[-1].startswith("ctl")]
            tri_list = ['.'.join(f) for f in lof if f[-1].startswith("tri")]
            egm_list = ['.'.join(f) for f in lof if f[-1].startswith("egm")]
            ctl = zf.open(ctl_list[0]).read()
            menu = zf.open(csv_list[0]).read()

            # tri, egm pairs should have the same name
            models = list()
            for tri,egm in zip(tri_list, egm_list):
                tri = zf.open(tri).read()
                egm = zf.open(egm).read()
                models.append((tri,egm))

            # catch endianness from facegen files extensions
            fg_ext = [f[-1] for f in lof if f[-1].startswith(("ctl","tri","egm"))]
            if all(ext.endswith("be") for ext in fg_ext):
                endian = "big"
            elif all(ext.endswith("le") for ext in fg_ext):
                endian = "little"
            else:
                endian = None
        return cls(ctl, menu, models, endian=endian)

    @property
    def values (self):
        return {k:s.value for k,s in self.sliders.items() if not s.debug_only}

    @property
    def labels (self):
        return {k:s.label for k,s in self.sliders.items() if not s.debug_only}

    @property
    def tabs (self):
        return {k:s.tab for k,s in self.sliders.items() if not s.debug_only}

    @property
    def face (self):
        return self.models[0]

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
                value = int(items[-1])
                slider = self.sliders[key]
                slider.value = slider.int2float(value)
        if len(csv) != 0:
            self.all_at_once = True
            self.update()

    def save_values (self, fname, *, sort=True):
        out = ""
        if not sort:
            to_save = [slider for key,slider in self.sliders.items() if not slider.debug_only]
        else:
            to_save = [self.sliders[key] for tab in self.menu.values() for key in tab]

        for slider in to_save:
            key = slider.facegen_id
            tab = slider.tab
            lab = slider.label
            val = slider.int_value
            out += f"{key:03d}, {tab}, {lab}, {val};\n"

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
        self.menu.clear()
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
                if tab not in self.menu:
                    self.menu[tab] = list()
                self.menu[tab].append(key)
            elif key==">>>":
                self.sequence = [int(k) for k in items]
                has_sequence = True
            elif key==">1<":
                gs_min, gs_max = items
                self.shape_sym_range = (float(gs_min), float(gs_max))
            elif key==">2<":
                ts_min, ts_max = items
                self.texture_sym_range = (float(ts_min), float(ts_max))
            elif key==">3<":
                ga_min, ga_max = items
                self.shape_asym_range = (float(ga_min), float(ga_max))
            elif key==">4<":
                ta_min, ta_max = items
                self.texture_asym_range = (float(ta_min), float(ta_max))

        if len(self.menu) == 0:
            for key,slider in self.sliders.items():
                tab = slider.tab
                if tab not in self.menu:
                    self.menu[tab] = list()
                self.menu[tab].append(key)

        self.all_at_once = has_sequence
        self.reset_values()
        if not has_sequence:
            self.sequence = [key for tab in self.menu.values() for key in tab]

    def update (self):
        sam = self.models[0]
        if self.all_at_once:
            self.set_zero(sam)
            for key in self.sequence:
                value = self.sliders[key].value
                self.set_control(key, value, sam)
            self.clip_data(sam)
        else:
            self.clip_data(sam)
            for key,slider in self.sliders.items():
                slider.value = self.get_control(key, sam)

    def toggle_sequence (self, new=None):
        if self.all_at_once == new:
            return
        self.all_at_once = not self.all_at_once
        self.update()

    def set_slider (self, key, value):
        slider = self.sliders[key]
        if isinstance(value, int):
            value = slider.int2float(value)
        if self.all_at_once:
            slider.value = value
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
        for key,slider in self.sliders.items():
            slider.value = self.get_control(key) if not slider.debug_only else sum(slider.float_range)/2.0
        self.update()

    def reset_sliders (self):
        self.sliders.clear()
        nadd = {100: self.LGS, 200: self.LTS, 300: self.LGA, 400: self.LTA}
        for key in SETMAP:
            n = nadd.get(key, 1)
            for idx in range(n):
                self.sliders[key+idx] = CharacterSlider(self, key+idx)

    def get_control (self, key, sam=None):
        if sam is None: sam = self.models[0]
        i = key % 100
        if key >= 100:
            k = (key//100)*100
            return GETMAP[k](self, i, sam)
        else:
            return GETMAP[i](self, sam)

    def set_control (self, key, value, sam=None):
        if sam is None: sam = self.models[0]
        i = key % 100
        if key >= 100:
            k = (key//100)*100
            SETMAP[k](self, i, value, sam)
        else:
            SETMAP[i](self, value, sam)


class CharacterSlider:
    def __init__ (self, parent, key):
        self.parent = parent
        self.facegen_id = key
        self.debug_only = True
        self.debug_tab = DFTABS.get((key//100)*100)
        if key < 100:
            self.debug_label = DBLABELS.get(key)
            self.float_range = DFRANGES.get((key//10)*10)
        else:
            k = (key//100)*100
            idx = key % 100
            self.debug_label = getattr(parent, DBLABELS.get(k))[idx]
            self.float_range = DFRANGES.get(k)
        self.int_range = (0, 255)
        self.tab = self.debug_tab
        self.label = self.debug_label
        self.value = sum(self.float_range)/2.0

    @property
    def available_range (self):
        int_min, int_max = self.int_range
        af_range = self.int2float(int_min), self.int2float(int_max)
        return min(af_range), max(af_range)

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

    # header comments
    content = content.lstrip()
    while content.startswith('#'):
        endl = content.find('\n')
        content = content[endl+1:].lstrip()

    # rows separated by ;
    rows = list(map(lambda s: s.strip(), content.split(';')))
    rows = [r for r in rows if len(r)>0]

    # cells separated by ,
    for r in rows:
        cells = tuple(map(lambda s: s.strip(), r.split(',')))
        key = str(cells[0])
        if key.isdigit():
            out[int(key)] = cells[1:]
        else:
            out[key] = cells[1:]

    return out


DFTABS = {
0: "[Generate]",
100: "[Shape Sym.]",
200: "[Texture Sym.]",
300: "[Shape Asym.]",
400: "[Texture Asym.]"}

DFRANGES = {
10: (15.0, 60.0),
20: (-4.0, 4.0),
30: (0.0, 2.0),
40: (0.0, 2.0),
100: (-10.0, 10.0),
200: (-10.0, 10.0),
300: (-10.0, 10.0),
400: (-10.0, 10.0)}

DBLABELS = {
10: "Age",
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

GETMAP = {
10: FaceGenerator.get_age,
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

SETMAP = {
10: FaceGenerator.set_age_neutral,
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
