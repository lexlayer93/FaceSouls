from .models import FaceGenerator, FaceGenSAM


class CharacterCreator (FaceGenerator):
    def __init__ (self, ctl, menu=None, models=None, *, preset=None, endian=None):
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
            self.load_models(*models)

        self.load_menu(menu) # sets all_at_once

        if preset is None:
            return
        if self.all_at_once:
            self.load_values(preset)
        else:
            self.load_data(preset, endian=endian)

    def load_models (self, *models, endian=None):
        self.models = []
        for sam in models:
            if not isinstance(sam, FaceGenSAM):
                sam = FaceGenSAM(*sam, endian=endian)
            self.fix_model(sam)
            self.models.append(sam)
        self.sync_models()

    def load_menu (self, fname):
        self.reset_sliders()
        self.tabs.clear()
        flag = False
        with open(fname, 'r') as f: # menu file
            rows = f.read().split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                key = cells[0]
                if key[0] != '#':
                    key = int(key)
                    tab, label, float_min, float_max, int_min, int_max = cells[1:]
                    slider = self.sliders[key]
                    slider.debug_only = False
                    slider.label = label
                    slider.float_range = (float(float_min), float(float_max))
                    slider.int_range = (int(int_min), int(int_max))
                    if tab not in self.tabs:
                        self.tabs[tab] = list()
                    self.tabs[tab].append(key)
                elif key=="###":
                    self.sequence = [int(c) for c in cells[1:]]
                    flag = True
                elif key=="##1":
                    self.shape_sym_range = (float(cells[1]), float(cells[2]))
                elif key=="##2":
                    self.texture_sym_range = (float(cells[1]), float(cells[2]))
                elif key=="##3":
                    self.shape_asym_range = (float(cells[1]), float(cells[2]))
                elif key=="##4":
                    self.texture_asym_range = (float(cells[1]), float(cells[2]))
                else:
                    pass
        self.reset_values()
        if flag:
            self.all_at_once = True
            self.apply_sequence()
        else:
            self.all_at_once = False
            self.update_values()
            self.sequence = [key for tab in self.tabs.values() for key in tab]

    def load_data (self, fname, *, endian=None):
        try:
            self.models[0].load_data(fname, endian=endian)
        except FileNotFoundError:
            pass
        else:
            self.sync_models()
            self.clip_data()
            self.update_values()
            self.all_at_once = False

    def save_data (self, fname, *, endian=None):
        self.models[0].save_data(fname, endian=endian)

    def load_values (self, fname):
        try:
            with open(fname, 'r') as f:
                rows = f.read().split(';')
                del rows[-1]
                for r in rows:
                    cells = list(map(lambda s: s.strip(), r.split(',')))
                    key, value = cells
                    key, value = int(key), int(value)
                    slider = self.sliders[int(key)]
                    value = slider.int2float(value)
                    self.values[key] = value
        except FileNotFoundError:
            pass
        else:
            self.apply_sequence()
            self.all_at_once = True

    def save_values (self, fname, *, sort=False):
        output = ""
        if not sort:
            for key,slider in self.sliders.items():
                if not slider.debug_only or key in self.sequence:
                    output += f"{key:03d}, {slider.int_value};\n"
        else:
            for tab in self.tabs.values():
                for key in tab:
                    key = int(key)
                    slider = self.sliders[key]
                    output += f"{key:03d},{slider.int_value}; "
                output += '\n'
        with open(fname, 'w') as f:
            f.write(output)

    def set_slider (self, key, value):
        if isinstance(value, int):
            value = self.sliders[key].int2float(value)

        self.values[key] = float(value)

        if self.all_at_once:
            self.apply_sequence()
        else:
            self.set_control(key, value)
            self.clip_data()
            self.update_values()

    def toggle_sequence (self, flag=None):
        if self.all_at_once == flag:
            return
        self.all_at_once = not self.all_at_once

        if self.all_at_once:
            self.apply_sequence()
        else:
            self.update_values()

    def apply_sequence (self, *, from_zero=True):
        sam = self.models[0]
        if from_zero: self.set_zero(sam)
        for key in self.sequence:
            value = self.values[key]
            self.set_control(key, value, sam)
            self.clip_data(sam)

    def update_values (self):
        src = self.models[0]
        for key in self.values:
            self.values[key] = self.get_control(key, src)

    def clip_data (self, sam=None):
        if sam is None: sam = self.models[0]
        sam.gs_data.clip(*self.shape_sym_range, sam.gs_data)
        sam.ts_data.clip(*self.texture_sym_range, sam.ts_data)
        sam.ga_data.clip(*self.shape_asym_range, sam.ga_data)
        sam.ta_data.clip(*self.texture_asym_range, sam.ta_data)

    def sync_models (self, src=None):
        if src is None: src = self.models[0]
        for sam in self.models:
            if sam == src: continue
            sam.gs_data = src.gs_data
            sam.ts_data = src.ts_data
            sam.ga_data = src.ga_data
            sam.ta_data = src.ta_data

    def reset_values (self):
        self.values.clear()
        for key,slider in self.sliders.items():
            self.values[key] = sum(slider.float_range)/2

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
