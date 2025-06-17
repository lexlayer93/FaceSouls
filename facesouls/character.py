from dataclasses import dataclass
from .models import FaceGenerator

@dataclass(init=False)
class CharacterSlider:
    label: str = ""
    value: float = 0.0
    float_range: tuple[float, float] = (-10.0, 10.0)
    int_range: tuple[int, int] = (0, 255)
    debug_label: str = ""
    debug_only: bool = True

    @property
    def int_value (self):
        return self.float2int(self.value)

    @property
    def available_float_range (self):
        int_min, int_max = self.int_range
        af_range = self.int2float(int_min), self.int2float(int_max)
        return min(af_range), max(af_range)

    def int2float (self, x):
        float_min, float_max = self.float_range
        return float_min + x*(float_max - float_min)/255

    def float2int (self, x):
        float_min, float_max = self.float_range
        return round(255*(x - float_min)/(float_max - float_min))

class CharacterCreator (FaceGenerator):
    def __init__ (self, ctl, menu=None, models=None, data=None, endian="little"):
        super().__init__(ctl, endian)
        self.all_at_once = False
        self.sliders = dict()
        self.tabs = dict()
        self.sequence = list()
        no_limit = (float("-inf"), float("inf"))
        self.shape_sym_range = no_limit
        self.texture_sym_range = no_limit
        self.shape_asym_range = no_limit
        self.texture_asym_range = no_limit
        self.load_menu(menu)
        if models is None: self.models = [self.test_model()]
        self.load_data(data, endian)

    def load_menu (self, fname=None):
        self.reset_sliders()

        if fname is None:
            return

        self.tabs.clear()
        with open(fname, 'r') as f: # menu file
            rows = f.read().split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                key = cells[0]
                if key[0] != '#':
                    fg_id, tab, label, float_min, float_max, int_min, int_max = cells
                    slider = self.sliders[fg_id]
                    slider.label = label
                    slider.float_range = (float(float_min), float(float_max))
                    slider.int_range = (int(int_min), int(int_max))
                    slider.debug_only = False
                    if tab not in self.tabs:
                        self.tabs[tab] = list()
                    self.tabs[tab].append(fg_id)
                elif key=="#Seq":
                    self.sequence = list.copy(cells[1:])
                    self.all_at_once = True
                elif key=="#GS":
                    self.shape_sym_range = (float(cells[1]), float(cells[2]))
                elif key=="#TS":
                    self.texture_sym_range = (float(cells[1]), float(cells[2]))
                elif key=="#GA":
                    self.shape_asym_range = (float(cells[1]), float(cells[2]))
                elif key=="#TA":
                    self.texture_asym_range = (float(cells[1]), float(cells[2]))
                else:
                    pass

    def load_data (self, fname, endian="little"):
        if self.all_at_once:
            if fname is not None:
                with open(fname, 'r') as f:
                    rows = f.read.split(';')
                    del rows[-1]
                    for r in rows:
                        cells = list(map(lambda s: s.strip(), r.split(',')))
                        fg_id, value = cells
                        slider = self.sliders[fg_id]
                        slider.value = slider.int2float(value)
            self.apply_sequence()
        else:
            sam.load_data(fname, endian)
            self.update_sliders()
        self.sync_models()

    def set_slider (self, fg_id, value):
        slider = self.sliders[fg_id]
        if isinstance(value, float):
            slider.value = value
        elif isinstance(value, int):
            slider.value = slider.int2float(value)

        sam = self.models[0]
        if self.all_at_once:
            self.set_zero(sam)
            self.apply_sequence(sam)
        else:
            value = slider.value
            self.set_control(fg_id, value, sam)
            self.update_sliders(sam)
        self.sync_models(sam)

    def get_values (self, fg_ids):
        return [self.sliders[fg_id].value for fg_id in fg_ids]

    def apply_sequence (self, sam=None, fg_ids=None):
        if sam is None: sam = self.models[0]
        if fg_ids is None: fg_ids = self.sequence
        for fg_id in fg_ids:
            value = self.sliders[fg_id].value
            self.set_control(fg_id, value, sam)

    def update_sliders (self, src=None):
        if src is None: src = self.models[0]
        for fg_id, slider in self.sliders.items():
            slider.value = self.get_control(fg_id, src)

    def sync_models (self, src=None):
        if src is None: src = self.models[0]
        for sam in self.models:
            sam.gs_data = src.gs_data.copy()
            sam.ts_data = src.ts_data.copy()
            sam.ga_data = src.ga_data.copy()
            sam.ta_data = src.ta_data.copy()

    def get_control (self, fg_id, sam):
        if fg_id == "Age":
            value = self.get_shape_age(sam)
        elif fg_id == "Gnd":
            value = self.get_shape_gender(sam)
        elif fg_id == "Car":
            value = self.get_shape_caricature_neutral(sam)
        elif fg_id[:3] == "LGS":
            idx = int(fg_id[3:])
            value = self.get_shape_sym_control(idx, sam)
        elif fg_id[:3] == "LTS":
            idx = int(fg_id[3:])
            value = self.get_texture_sym_control(idx, sam)
        elif fg_id == "Asy":
            value = self.get_shape_asymmetry(sam)
        elif fg_id[:3] == "LGA":
            idx = int(fg_id[3:])
            value = self.get_shape_asym_control(idx, sam)
        elif fg_id[:3] == "LTA":
            idx = int(fg_id[3:])
            value = self.get_texture_asym_control(idx, sam)
        else:
            return None
        return value

    def set_control (self, fg_id, value, sam):
        if fg_id == "Age":
            self.set_shape_age_neutral(value, sam)
            self.set_texture_age_neutral(value, sam)
            self.clip_shape_sym(*self.shape_sym_range, sam)
            self.clip_texture_sym(*self.texture_sym_range, sam)
        elif fg_id == "Gnd":
            self.set_shape_gender_neutral(value, sam)
            self.set_texture_gender_neutral(value, sam)
            self.clip_shape_sym(*self.shape_sym_range, sam)
            self.clip_texture_sym(*self.texture_sym_range, sam)
        elif fg_id == "Car":
            self.set_shape_caricature_neutral(value, sam)
            self.set_texture_caricature_neutral(value, sam)
            self.clip_shape_sym(*self.shape_sym_range, sam)
            self.clip_texture_sym(*self.texture_sym_range, sam)
        elif fg_id[:3] == "LGS":
            idx = int(fg_id[3:])
            self.set_shape_sym_control(idx, value, sam)
            self.clip_shape_sym(*self.shape_sym_range, sam)
        elif fg_id[:3] == "LTS":
            idx = int(fg_id[3:])
            self.set_texture_sym_control(idx, value, sam)
            self.clip_texture_sym(*self.texture_sym_range, sam)
        elif fg_id == "Asy":
            self.set_shape_asymmetry(value, sam)
            self.set_texture_asymmetry(value, sam)
            self.clip_shape_asym(*self.shape_asym_range, sam)
            self.clip_texture_asym(*self.texture_asym_range, sam)
        elif fg_id[:3] == "LGA":
            idx = int(fg_id[3:])
            self.set_shape_asym_control(idx, value, sam)
            self.clip_shape_asym(*self.shape_asym_range, sam)
        elif fg_id[:3] == "LTA":
            idx = int(fg_id[3:])
            self.set_texture_asym_control(idx, value, sam)
            self.clip_texture_asym(*self.texture_asym_range, sam)
        else:
            return None
        return value

    def reset_sliders (self):
        self.sliders.clear()
        for k in ("Age", "Gnd", "Car", "Asy"):
            self.sliders[k] = CharacterSlider()
        for i in range(self.LGS):
            self.sliders[f"LGS{i:02d}"] = CharacterSlider()
        for i in range(self.LTS):
            self.sliders[f"LTS{i:02d}"] = CharacterSlider()
        for i in range(self.LGA):
            self.sliders[f"LGA{i:02d}"] = CharacterSlider()
        for i in range(self.LTA):
            self.sliders[f"LTA{i:02d}"] = CharacterSlider()

        for fg_id, slider in self.sliders.items():
            if fg_id == "Age":
                slider.value = self.current_race.age_control.gs_offset
                slider.float_range = (15.0, 60.0)
                slider.debug_label = "Age"
            elif fg_id == "Gnd":
                slider.value = self.current_race.gnd_control.gs_offset
                slider.float_range = (-4.0, 4.0)
                slider.debug_label = "Gender"
            elif fg_id == "Car":
                slider.value = 0.0
                slider.float_range = (0.0, 2.0)
                slider.debug_label = "Caricature"
            elif fg_id == "Asy":
                slider.value = 0.0
                slider.float_range = (0.0, 2.0)
                slider.debug_label = "Asymmetry"
            elif fg_id[:3] == "LGS":
                idx = int(fg_id[3:])
                slider.debug_label = self.lgs_labels[idx]
            elif fg_id[:3] == "LTS":
                idx = int(fg_id[3:])
                slider.debug_label = self.lts_labels[idx]
            elif fg_id[:3] == "LGA":
                idx = int(fg_id[3:])
                slider.debug_label = self.lga_labels[idx]
            elif fg_id[:3] == "LTA":
                idx = int(fg_id[3:])
                slider.debug_label = self.lta_labels[idx]


