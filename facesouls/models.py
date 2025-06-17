import numpy as np
from PIL import Image
from .files import *

class FaceGenSSM:
    def __init__ (self, tri=None, egm=None, fg=None, endian="little"):
        self.load_shape_data(fg, endian)
        self.load_shape_model(tri, egm, endian)

    @property
    def vertices (self):
        try:
            return self.vertices_default + np.dot(self.gs_deltas, self.gs_data) + np.dot(self.ga_deltas, self.ga_data)
        except:
            return None

    def load_shape_model (self, tri, egm, endian="little"):
        if isinstance(tri, str):
            tri = FaceGenTRI(tri, endian)
            quads = np.array(tri.mesh_quads, dtype=np.int32)
            triangles1 = np.array(tri.mesh_triangles, dtype=np.int32)
            triangles2 = np.delete(quads, 1, axis=1)
            triangles3 = np.delete(quads, 3, axis=1)
            self.triangles = np.concatenate((triangles1, triangles2, triangles3), axis=0)
            self.vertices_default = np.array(tri.mesh_vertices, dtype=np.float32)

        if isinstance(egm, str):
            egm = FaceGenEGM(egm, endian="little")
            self.GS = np.uint32(egm.GS)
            self.GA = np.uint32(egm.GA)
            self.gs_deltas = [np.array(egm.gs_deltas[i])*egm.gs_scales[i] for i in range(egm.GS)]
            self.gs_deltas = np.transpose(self.gs_deltas, (1,2,0)).astype(np.float32)
            self.ga_deltas = [np.array(egm.ga_deltas[i])*egm.ga_scales[i] for i in range(egm.GA)]
            self.ga_deltas = np.transpose(self.ga_deltas, (1,2,0)).astype(np.float32)

        return tri, egm

    def load_shape_data (self, fg, endian="little"):
        if isinstance(fg, str):
            fg = FaceGenFG(fg, endian)
        else:
            fg = FaceGenFG()
        self.GS = np.uint32(fg.GS)
        self.GA = np.uint32(fg.GA)
        self.gs_data = np.array(fg.gs_data, dtype=np.float32)/1000
        self.ga_data = np.array(fg.ga_data, dtype=np.float32)/1000
        return fg


class FaceGenSTM:
    def __init__ (self, bmp=None, egt=None, fg=None, endian="little"):
        self.load_texture_data(fg, endian)
        self.load_texture_model(bmp, egt, endian)

    @property
    def pixels (self):
        try:
            return self.pixels_default + np.dot(self.ts_deltas, self.ts_data) + np.dot(self.ta_deltas, self.ta_data)
        except:
            return None

    def load_texture_model (self, bmp, egt, endian="little"):
        if isinstance(bmp, str):
            img = Image.open(bmp)
            self.pixels_default = np.asarray(img, dtype=np.float32)

        if isinstance(egt, str):
            egt = FaceGenEGT(egt, endian)
            self.TS = np.uint32(egt.TS)
            self.TA = np.uint32(egt.TA)
            ts_deltas = [np.array(egt.ts_deltas[i])*egt.ts_scales[i] for i in range(egt.TS)]
            ts_deltas = ts_deltas.transpose(1,2,0).astype(np.float32)
            self.ts_deltas = ts_deltas.reshape(egt.image_height, egt.image_width, 3, egt.TS)
            ta_deltas = [np.array(egt.ta_deltas[i])*egt.ta_scales[i] for i in range(egt.TA)]
            ta_deltas = ta_deltas.transpose(1,2,0).astype(np.float32)
            self.ta_deltas = ta_deltas.reshape(egt.image_height, egt.image_width, 3, egt.TA)

        return bmp, egt

    def load_texture_data (self, fg, endian="little"):
        if isinstance(fg, str):
            fg = FaceGenFG(fg, endian)
        else:
            fg = FaceGenFG()
        self.TS = np.uint32(fg.TS)
        self.TA = np.uint32(fg.TA)
        self.ts_data = np.array(fg.ts_data, dtype=np.float32)/1000
        self.ta_data = np.array(fg.ta_data, dtype=np.float32)/1000
        return fg


class FaceGenSAM (FaceGenSSM, FaceGenSTM):
    def __init__ (self, tri=None, egm=None, bmp=None, egt=None, fg=None, endian="little"):
        FaceGenSSM.__init__(self, tri, egm, fg, endian)
        FaceGenSTM.__init__(self, bmp, egt, fg, endian)


class FaceGenerator:
    def __init__ (self, ctl, endian="little"):
        self.race = "All"
        self.load(ctl, endian)

    def load (self, ctl_fname, endian="little"):
        ctl = FaceGenCTL(ctl_fname, endian)
        self.GS = np.uint32(ctl.GS)
        self.GA = np.uint32(ctl.GA)
        self.TS = np.uint32(ctl.TS)
        self.TA = np.uint32(ctl.TA)
        self.LGS = np.uint32(ctl.LGS)
        self.LGA = np.uint32(ctl.LGA)
        self.LTS = np.uint32(ctl.LTS)
        self.LTA = np.uint32(ctl.LTA)
        self.lgs_coeffs = np.array([c.coeff for c in ctl.lgs_controls], dtype=np.float32)
        self.lga_coeffs = np.array([c.coeff for c in ctl.lga_controls], dtype=np.float32)
        self.lts_coeffs = np.array([c.coeff for c in ctl.lts_controls], dtype=np.float32)
        self.lta_coeffs = np.array([c.coeff for c in ctl.lta_controls], dtype=np.float32)
        self.lgs_labels = [c.label for c in ctl.lgs_controls]
        self.lga_labels = [c.label for c in ctl.lga_controls]
        self.lts_labels = [c.label for c in ctl.lts_controls]
        self.lta_labels = [c.label for c in ctl.lta_controls]
        self.races = ctl.races
        for race in self.races.values():
            race.age_control.gs_coeff = np.array(race.age_control.gs_coeff, dtype=np.float32)
            race.age_control.ts_coeff = np.array(race.age_control.ts_coeff, dtype=np.float32)
            race.age_control.gs_offset = np.float32(race.age_control.gs_offset)
            race.age_control.ts_offset = np.float32(race.age_control.ts_offset)
            race.gnd_control.gs_coeff = np.array(race.gnd_control.gs_coeff, dtype=np.float32)
            race.gnd_control.ts_coeff = np.array(race.gnd_control.ts_coeff, dtype=np.float32)
            race.gnd_control.gs_offset = np.float32(race.gnd_control.gs_offset)
            race.gnd_control.ts_offset = np.float32(race.gnd_control.ts_offset)
            race.gs_mean = np.array(race.gs_mean, dtype=np.float32)
            race.ts_mean = np.array(race.ts_mean, dtype=np.float32)
            race.gs_density = np.array(race.gs_density, dtype=np.float32).reshape(self.GS, self.GS)
            race.ts_density = np.array(race.ts_density, dtype=np.float32).reshape(self.TS, self.TS)
            race.combined_density = np.array(race.combined_density, dtype=np.float32).reshape(self.GS+self.TS, self.GS+self.TS)
            for morph in race.morph_controls.values():
                morph.gs_coeff = np.array(morph.gs_coeff, dtype=np.float32)
                morph.ts_coeff = np.array(morph.ts_coeff, dtype=np.float32)
                morph.offset = np.float32(morph.offset)

            vage = race.age_control.gs_coeff
            vgnd = race.gnd_control.gs_coeff
            cov = np.stack([vage, vgnd])
            cov = np.dot(cov, cov.T)
            race._ag_shape_covinv = np.linalg.inv(cov)

            uage = vage/np.linalg.norm(vage)
            ugnd = vgnd - uage * np.dot(uage, vgnd)
            ugnd = ugnd/np.linalg.norm(ugnd)
            uage = uage.reshape((uage.size,1))
            ugnd = ugnd.reshape((ugnd.size,1))
            race._ag_shape_proj = np.dot(uage, uage.T) + np.dot(ugnd, ugnd.T)

            vage = race.age_control.ts_coeff
            vgnd = race.gnd_control.ts_coeff
            cov = np.stack([vage, vgnd])
            cov = np.dot(cov, cov.T)
            race._ag_texture_covinv = np.linalg.inv(cov)

            uage = vage/np.linalg.norm(vage)
            ugnd = vgnd - uage * np.dot(uage, vgnd)
            ugnd = ugnd/np.linalg.norm(ugnd)
            uage = uage.reshape((uage.size,1))
            ugnd = ugnd.reshape((ugnd.size,1))
            race._ag_texture_proj = np.dot(uage, uage.T) + np.dot(ugnd, ugnd.T)
        self.set_race("All")
        return ctl

    def set_race (self, r):
        self.current_race = self.races[r]

    def set_zero (self, sam):
        self.set_shape_zero(sam)
        self.set_texture_zero(sam)

    def set_shape_zero (self, ssm):
        ssm.gs_data.fill(0.0)
        ssm.ga_data.fill(0.0)

    def set_texture_zero (self, stm):
        stm.ts_data.fill(0.0)
        stm.ta_data.fill(0.0)

    def clip_shape_sym (self, vmin, vmax, ssm):
        ssm.gs_data.clip(vmin, vmax, ssm.gs_data)

    def clip_shape_asym (self, vmin, vmax, ssm):
        ssm.ga_data.clip(vmin, vmax, ssm.ga_data)

    def clip_texture_sym (self, vmin, vmax, stm):
        stm.gs_data.clip(vmin, vmax, stm.gs_data)

    def clip_texture_asym (self, vmin, vmax, stm):
        stm.ga_data.clip(vmin, vmax, stm.ga_data)

    def get_shape_sym_control (self, idx, ssm):
        return np.dot(self.lgs_coeffs[idx,:], ssm.gs_data)

    def set_shape_sym_control (self, idx, value, ssm):
        value0 = self.get_shape_sym_control(idx, ssm)
        ssm.gs_data += (value - value0) * self.lgs_coeffs[idx,:]

    def get_shape_asym_control (self, idx, ssm):
        return np.dot(self.lga_coeffs[idx,:], ssm.ga_data)

    def set_shape_asym_control (self, idx, value, ssm):
        value0 = self.get_shape_asym_control(idx, ssm)
        ssm.ga_data += (value - value0) * self.lga_coeffs[idx,:]

    def get_texture_sym_control (self, idx, stm):
        return np.dot(self.lts_coeffs[idx,:], stm.ts_data)

    def set_texture_sym_control (self, idx, value, stm):
        value0 = self.get_texture_sym_control(idx, stm)
        stm.ts_data += (value - value0) * self.lts_coeffs[idx,:]

    def get_texture_asym_control (self, idx, stm):
        return np.dot(self.lta_coeffs[idx,:], stm.ta_data)

    def set_texture_asym_control (self, idx, value, stm):
        value0 = self.get_texture_asym_control(idx, stm)
        stm.ta_data += (value - value0) * self.lta_coeffs[idx,:]

    def get_shape_age (self, ssm):
        control = self.current_race.age_control
        return np.dot(control.gs_coeff, ssm.gs_data) + control.gs_offset

    def set_shape_age (self, value, ssm):
        control = self.current_race.age_control
        value0 = self.get_shape_age(ssm)
        ac = control.gs_coeff
        factor = (value - value0)/np.dot(ac, ac)
        ssm.gs_data += factor * ac

    def set_shape_age_neutral (self, value, ssm):
        ac = self.current_race.age_control.gs_coeff
        gc = self.current_race.gnd_control.gs_coeff
        value0 = self.get_shape_age(ssm)
        af, gf = np.dot(self.current_race._ag_shape_covinv, (value-value0, 0))
        ssm.gs_data += af*ac + gf*gc

    def get_texture_age (self, stm):
        control = self.current_race.age_control
        return np.dot(control.ts_coeff, stm.ts_data) + control.ts_offset

    def set_texture_age (self, value, stm):
        control = self.current_race.age_control
        value0 = self.get_texture_age(stm)
        ac = control.ts_coeff
        af = (value - value0)/np.dot(ac, ac)
        stm.ts_data += af*ac

    def set_texture_age_neutral (self, value, stm):
        ac = self.current_race.age_control.ts_coeff
        gc = self.current_race.gnd_control.ts_coeff
        value0 = self.get_shape_age(stm)
        af, gf = np.dot(self.current_race._ag_texture_covinv, (value-value0, 0))
        stm.ts_data += af*ac + gf*gc

    def get_shape_gender (self, ssm):
        control = self.current_race.gnd_control
        return np.dot(control.gs_coeff, ssm.gs_data) + control.gs_offset

    def set_shape_gender (self, value, ssm):
        control = self.current_race.gnd_control
        value0 = self.get_shape_gender(ssm)
        ac = control.gs_coeff
        factor = (value - value0)/np.dot(ac, ac)
        ssm.gs_data += factor * ac

    def set_shape_gender_neutral (self, value, ssm):
        ac = self.current_race.age_control.gs_coeff
        gc = self.current_race.gnd_control.gs_coeff
        value0 = self.get_shape_gender(ssm)
        af, gf = np.dot(self.current_race._ag_shape_covinv, (0, value-value0))
        ssm.gs_data += af*ac + gf*gc

    def get_texture_gender (self, stm):
        control = self.current_race.gnd_control
        return np.dot(control.ts_coeff, stm.ts_data) + control.ts_offset

    def set_texture_gender (self, value, stm):
        control = self.current_race.gnd_control
        value0 = self.get_texture_gender(stm)
        ac = control.ts_coeff
        af = (value - value0)/np.dot(ac, ac)
        stm.ts_data += af*ac

    def set_texture_gender_neutral (self, value, stm):
        ac = self.current_race.age_control.ts_coeff
        gc = self.current_race.gnd_control.ts_coeff
        value0 = self.get_texture_gender(stm)
        af, gf = np.dot(self.current_race._ag_shape_covinv, (0, value-value0))
        stm.ts_data += af*ac + gf*gc

    def get_shape_caricature (self, ssm):
        race = self.current_race
        dev = ssm.gs_data - race.gs_mean
        q = np.dot(race.gs_density, dev)
        return np.linalg.norm(q)

    def set_shape_caricature (self, value, ssm):
        race = self.current_race
        value = max(0.01, value)
        value0 = self.get_shape_caricature(ssm)
        if value0 > 0:
            dev = ssm.gs_data - race.gs_mean
            ssm.gs_data = race.gs_mean + value/value0*dev

    def get_shape_caricature_neutral (self, ssm):
        race = self.current_race
        dev = ssm.gs_data - race.gs_mean
        ag_dev = race._ag_shape_proj.dot(dev)
        q = np.dot(race.gs_density, dev - ag_dev)
        return np.linalg.norm(q)/np.sqrt(ssm.GS-2, dtype=np.float32)

    def set_shape_caricature_neutral (self, value, ssm):
        race = self.current_race
        value = max(0.01, value)
        value0 = self.get_shape_caricature_neutral(ssm)
        if value0 > 0:
            dev = ssm.gs_data - race.gs_mean
            ag_dev = race._ag_shape_proj.dot(dev)
            ssm.gs_data = race.gs_mean + value/value0*(dev - ag_dev) + ag_dev

    def get_texture_caricature (self, stm):
        race = self.current_race
        dev = stm.ts_data - race.ts_mean
        q = np.dot(race.ts_density, dev)
        return np.linalg.norm(q)

    def set_texture_caricature (self, value, stm):
        race = self.current_race
        value = max(0.01, value)
        value0 = self.get_texture_caricature(stm)
        if value0 > 0:
            dev = stm.ts_data - race.ts_mean
            stm.ts_data = race.ts_mean + value/value0*dev

    def get_texture_caricature_neutral (self, stm):
        race = self.current_race
        dev = stm.ts_data - race.ts_mean
        ag_dev = race._ag_texture_proj.dot(dev)
        q = np.dot(race.ts_density, dev - ag_dev)
        return np.linalg.norm(q)/np.sqrt(stm.TS-2, dtype=np.float32)

    def set_texture_caricature_neutral (self, value, stm):
        race = self.current_race
        value = max(0.01, value)
        value0 = self.get_texture_caricature_neutral(stm)
        if value0 > 0:
            dev = stm.ts_data - race.ts_mean
            ag_dev = race._ag_texture_proj.dot(dev)
            stm.ts_data = race.ts_mean + value/value0*(dev - ag_dev) + ag_dev

    def get_shape_asymmetry (self, ssm):
        return np.linalg.norm(ssm.ga_data)/np.sqrt(ssm.GA, dtype=np.float32)

    def set_shape_asymmetry (self, value, ssm):
        value0 = self.get_shape_asymmetry(ssm)
        if value0 > 0:
            ssm.ga_data *= value/value0

    def get_texture_asymmetry (self, stm):
        return np.linalg.norm(stm.ta_data)/np.sqrt(stm.TA, dtype=np.float32)

    def set_texture_asymmetry (self, value, stm):
        value0 = self.get_texture_asymmetry(stm)
        if value0 > 0:
            ssm.ta_data *= value/value0

    def get_race_morph(self, race, sam):
        control = self.current_race.morph[race]
        shape_term = np.dot(control.gs_coeff, sam.gs_data)
        texture_term = np.dot(control.ts_coeff, sam.ts_data)
        return shape_term + texture_term + control.offset

    def set_race_morph(self, race, value, sam):
        control = self.current_race.morph[race]
        value0 = self.get_race_morph(race, sam)
        sc2 = np.dot(control.gs_coeff, control.gs_coeff)
        tc2 = np.dot(control.ts_coeff, control.ts_coeff)
        k = (value - value0)/(sc2 + tc2)
        sam.gs_data += k * control.gs_coeff
        sam.ts_data += k * control.ts_coeff

    def test_model (self):
        sam = FaceGenSAM.__new__(FaceGenSAM)
        sam.GS, sam.GA = self.GS, self.GA
        sam.TS, sam.TA = self.TS, self.TA
        sam.gs_data = np.zeros(self.GS, dtype=np.float32)
        sam.ts_data = np.zeros(self.TS, dtype=np.float32)
        sam.ga_data = np.zeros(self.GA, dtype=np.float32)
        sam.ta_data = np.zeros(self.TA, dtype=np.float32)
        return sam
