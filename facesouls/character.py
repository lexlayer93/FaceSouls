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
        self.models = models or [self.test_model()]
        self.load_menu(menu)
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
        if fname is None or isinstance(fname,str) and fname.split('.')[-1] == "fg":
            self.all_at_once = False
            self.models[0].load_data(fname, endian)
            self.sync_models()
            self.update_sliders()
        else:
            self.all_at_once = True
            with open(fname, 'r') as f:
                rows = f.read.split(';')
                del rows[-1]
                for r in rows:
                    cells = list(map(lambda s: s.strip(), r.split(',')))
                    fg_id, value = cells
                    slider = self.sliders[fg_id]
                    slider.value = slider.int2float(value)
            self.sync_models()


    def load_param (self, fname):
        with open(fname, 'r') as f:
            rows = f.read.split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                fg_id, value = cells
                slider = self.sliders[fg_id]
                slider.value = slider.int2float(value)
        self.apply_sequence()
        self.all_at_once = True

    def set_slider (self, fg_id, value):
        slider = self.sliders[fg_id]

        if isinstance(value, float):
            slider.value = value
        elif isinstance(value, int):
            slider.value = slider.int2float(value)

        if self.all_at_once:
            self.set_sequence()
        else:
            for sam in self.models:
                self.set_control(fg_id, slider.value, sam)
            self.clip_data()
            self.update_sliders()

    def get_sequence (self, sequence=None):
        if sequence is None: sequence = self.sequence
        return [self.sliders[fg_id].value for fg_id in sequence]

    def set_sequence (self, sequence=None, values=None, models=None):
        if sequence is None: sequence = self.sequence
        if values is None: values = self.get_sequence(sequence)
        if models is None: models = self.models
        if not isinstance(models, list): models = [models]
        for sam in models:
            self.set_shape_zero(sam)
            self.set_texture_zero(sam)
            for i,fg_id in enumerate(sequence):
                value = values[i]
                self.set_control(fg_id, value, sam)
                self.clip_data()

    def update_sliders (self, model=None):
        sam = model or self.models[0]
        for fg_id, slider in self.sliders.items():
            slider.value = self.get_control(fg_id, sam)

    def sync_models (self, model=None):
        sam0 = model or self.models[0]
        if self.all_at_once:
            self.set_sequence()
        elif len(self.models) > 1:
            for sam in self.models[1:]:
                sam.gs_data = sam0.gs_data.copy()
                sam.ts_data = sam0.ts_data.copy()
                sam.ga_data = sam0.ga_data.copy()
                sam.ta_data = sam0.ta_data.copy()

    def clip_data (self):
        for sam in self.models:
            sam.gs_data.clip(*self.shape_sym_range, sam.gs_data)
            sam.ts_data.clip(*self.texture_sym_range, sam.ts_data)
            sam.ga_data.clip(*self.shape_asym_range, sam.ga_data)
            sam.ta_data.clip(*self.texture_asym_range, sam.ta_data)

    def set_control (self, fg_id, value, sam):
        if fg_id == "Age":
            self.set_shape_age_neutral(value, sam)
            self.set_texture_age_neutral(value, sam)
        elif fg_id == "Gnd":
            self.set_shape_gender_neutral(value, sam)
            self.set_texture_gender_neutral(value, sam)
        elif fg_id == "Car":
            self.set_shape_caricature_neutral(value, sam)
            self.set_texture_caricature_neutral(value, sam)
        elif fg_id[:3] == "LGS":
            idx = int(fg_id[3:])
            self.set_shape_sym_control(idx, value, sam)
        elif fg_id[:3] == "LTS":
            idx = int(fg_id[3:])
            self.set_texture_sym_control(idx, value, sam)
        elif fg_id == "Asy":
            self.set_shape_asymmetry(value, sam)
            self.set_texture_asymmetry(value, sam)
        elif fg_id[:3] == "LGA":
            idx = int(fg_id[3:])
            self.set_shape_asym_control(idx, value, sam)
        elif fg_id[:3] == "LTA":
            idx = int(fg_id[3:])
            self.set_texture_asym_control(idx, value, sam)
        return value

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


