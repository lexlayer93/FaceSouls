import numpy as np
from PIL import Image
from .files import *

__all__ = [
    "FaceGenSSM",
    "FaceGenSTM",
    "FaceGenSAM",
    "FaceGenerator"
    ]

class FaceGenSSM:
    def __init__ (self, tri=None, egm=None, fg=None, *, endian=None):
        if tri is not None and egm is not None:
            self.load_shape_model(tri, egm, endian=endian)
        else:
            self.GS = 50
            self.GA = 30
            self.geo_basis_version = 0
        if fg is not None:
            self.load_shape_data(fg, endian=endian)
        else:
            self.gs_data = np.zeros(self.GS, dtype=np.float32)
            self.ga_data = np.zeros(self.GA, dtype=np.float32)

    @property
    def vertices (self):
        return self.vertices0 + self.gs_deltas.dot(self.gs_data) + self.ga_deltas.dot(self.ga_data)

    def copy (self, *, to=None):
        if to is None: to = FaceGenSSM.__new__(FaceGenSSM)
        try:
            to.vertices0 = self.vertices0
            to.triangles = self.triangles
            to.quads = self.quads
            to.triangles_only = self.triangles_only
            to.uv_vertices = self.uv_vertices
            to.uv_triangles = self.uv_triangles
            to.uv_quads = self.uv_quads
            to.uv_triangles_only = self.uv_triangles_only
        except AttributeError:
            pass
        try:
            to.gs_deltas = self.gs_deltas
            to.ga_deltas = self.ga_deltas
        except AttributeError:
            pass
        to.geo_basis_version = self.geo_basis_version
        to.GS, to.GA = self.GS, self.GA
        to.gs_data = self.gs_data.copy()
        to.ga_data = self.ga_data.copy()
        return to

    def copy_shape_data (self, src):
        np.copyto(self.gs_data, src.gs_data)
        np.copyto(self.ga_data, src.ga_data)

    def load_shape_model (self, tri, egm, *, endian=None):
        if not isinstance(tri, FaceGenTRI):
            tri = FaceGenTRI(tri, endian=endian)
        self.vertices0 = np.array(tri.vertices, dtype=np.float32)
        self.triangles = np.array(tri.triangles, dtype=np.uint32)
        self.quads = np.array(tri.quads, dtype=np.uint32)
        self.uv_vertices = np.array(tri.uv_vertices, dtype=np.float32)
        self.uv_triangles = np.array(tri.uv_triangles, dtype=np.uint32)
        self.uv_quads = np.array(tri.uv_quads, dtype=np.uint32)
        if tri.num_quads > 0:
            triangles1 = self.triangles
            triangles2 = np.delete(self.quads, 1, axis=1)
            triangles3 = np.delete(self.quads, 3, axis=1)
            self.triangles_only = np.concatenate((triangles1, triangles2, triangles3), axis=0)
            triangles1 = self.uv_triangles
            triangles2 = np.delete(self.uv_quads, 1, axis=1)
            triangles3 = np.delete(self.uv_quads, 3, axis=1)
            self.uv_triangles_only = np.concatenate((triangles1, triangles2, triangles3), axis=0)
        else:
            self.triangles_only = self.triangles
            self.uv_triangles_only = self.uv_triangles

        if not isinstance(egm, FaceGenEGM):
            egm = FaceGenEGM(egm, endian=endian)
        self.geo_basis_version = egm.geo_basis_version
        self.GS = np.uint32(egm.GS)
        self.GA = np.uint32(egm.GA)
        gs_deltas = [np.array(egm.gs_deltas[i])*egm.gs_scales[i] for i in range(egm.GS)]
        self.gs_deltas = np.transpose(gs_deltas, (1,2,0)).astype(np.float32)
        ga_deltas = [np.array(egm.ga_deltas[i])*egm.ga_scales[i] for i in range(egm.GA)]
        self.ga_deltas = np.transpose(ga_deltas, (1,2,0)).astype(np.float32)

        return tri, egm

    def load_shape_data (self, fg, *, endian=None):
        if not isinstance(fg, FaceGenFG):
            fg = FaceGenFG(fg, endian=endian)
        self.gs_data = np.array(fg.gs_data, dtype=np.float32)/1000
        self.ga_data = np.array(fg.ga_data, dtype=np.float32)/1000
        return fg

    def dump_obj (self, *args):
        return FaceGenTRI.dump_obj(self, *args)


class FaceGenSTM:
    def __init__ (self, bmp=None, egt=None, fg=None, *, endian=None):
        if bmp is not None and egt is not None:
            self.load_texture_model(bmp, egt, endian=endian)
        else:
            self.TS = 50
            self.TA = 0
            self.tex_basis_version = 0
        if fg is not None:
            self.load_texture_data(fg, endian=endian)
        else:
            self.ts_data = np.zeros(self.TS, dtype=np.float32)
            self.ta_data = np.zeros(self.TA, dtype=np.float32)

    @property
    def pixels (self):
        return self.pixels0 + self.ts_deltas.dot(self.ts_data) + self.ta_deltas.dot(self.ta_data)

    def copy (self, *, to=None):
        if to is None: to = FaceGenSTM.__new__(FaceGenSTM)
        try:
            to.pixels0 = self.pixels0
        except AttributeError:
            pass
        try:
            to.ts_deltas = self.ts_deltas
            to.ta_deltas = self.ta_deltas
        except AttributeError:
            pass
        to.tex_basis_version = self.tex_basis_version
        to.TS, to.TA = self.TS, self.TA
        to.ts_data = self.ts_data.copy()
        to.ta_data = self.ta_data.copy()
        return to

    def copy_texture_data (self, src):
        np.copyto(self.ts_data, src.ts_data)
        np.copyto(self.ta_data, src.ta_data)

    def load_texture_model (self, bmp, egt, *, endian=None):
        img = Image.open(bmp)
        self.pixels0 = np.asarray(img, dtype=np.float32)

        if not isinstance(egt, FaceGenEGT):
            egt = FaceGenEGT(egt, endian=endian)
        self.tex_basis_version = egt.tex_basis_version
        self.TS = np.uint32(egt.TS)
        self.TA = np.uint32(egt.TA)
        ts_deltas = [np.array(egt.ts_deltas[i])*egt.ts_scales[i] for i in range(egt.TS)]
        ts_deltas = np.transpose(ts_deltas,(1,2,0)).astype(np.float32)
        self.ts_deltas = ts_deltas.reshape(egt.image_height, egt.image_width, 3, egt.TS)
        ta_deltas = [np.array(egt.ta_deltas[i])*egt.ta_scales[i] for i in range(egt.TA)]
        ta_deltas = np.transpose(ta_deltas,(1,2,0)).astype(np.float32)
        self.ta_deltas = ta_deltas.reshape(egt.image_height, egt.image_width, 3, egt.TA)
        egt = None

        return bmp, egt

    def load_texture_data (self, fg, *, endian=None):
        if not isinstance(fg, FaceGenFG):
            fg = FaceGenFG(fg, endian=endian)
        self.ts_data = np.array(fg.ts_data, dtype=np.float32)/1000
        self.ta_data = np.array(fg.ta_data, dtype=np.float32)/1000
        return fg


class FaceGenSAM (FaceGenSSM, FaceGenSTM):
    def __init__ (self, tri=None, egm=None, bmp=None, egt=None, fg=None, *, endian=None):
        FaceGenSSM.__init__(self, tri, egm, fg, endian=endian)
        FaceGenSTM.__init__(self, bmp, egt, fg, endian=endian)

    def copy (self, *, to=None):
        if to is None: to = FaceGenSAM.__new__(FaceGenSAM)
        FaceGenSSM.copy(self, to=to)
        FaceGenSTM.copy(self, to=to)
        return to

    def load_model (self, tri=None, egm=None, bmp=None, egt=None, *, endian=None):
        self.load_shape_model(tri, egm, endian=endian)
        self.load_texture_model(bmp, egt, endian=endian)

    def load_data (self, fg, *, endian=None):
        fg = self.load_shape_data(fg, endian=endian)
        self.load_texture_data(fg)

    def copy_data (self, src):
        self.copy_shape_data(src)
        self.copy_texture_data(src)

    def save_data (self, fname, *, endian=None):
        fg = FaceGenFG()
        fg.geo_basis_version = int(self.geo_basis_version)
        fg.tex_basis_version = int(self.tex_basis_version)
        fg.GS, fg.GA = int(self.GS), int(self.GA)
        fg.TS, fg.TA = int(self.TS), int(self.TA)
        fg.gs_data = (1000*self.gs_data).astype(int).tolist()
        fg.ga_data = (1000*self.ga_data).astype(int).tolist()
        fg.ts_data = (1000*self.ts_data).astype(int).tolist()
        fg.ta_data = (1000*self.ta_data).astype(int).tolist()
        fg.detail_texture_flag = 0
        fg.detail_image = b''
        fg.save(fname, endian=endian)


class FaceGenerator:
    def __init__ (self, ctl, *, endian=None):
        self.load(ctl, endian=endian)

    def load (self, ctl_fname, *, endian=None):
        ctl = FaceGenCTL(ctl_fname, endian=endian)
        self.geo_basis_version = ctl.geo_basis_version
        self.tex_basis_version = ctl.tex_basis_version
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

    def get_age (self, sam):
        return self.get_shape_age(sam)

    def set_age (self, value, sam):
        self.set_shape_age(value, sam)
        self.set_texture_age(value, sam)

    def set_age_neutral (self, value, sam):
        self.set_shape_age_neutral(value, sam)
        self.set_texture_age_neutral(value, sam)

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

    def get_gender (self, sam):
        return self.get_shape_gender(sam)

    def set_gender (self, value, sam):
        self.set_shape_gender(value, sam)
        self.set_texture_gender(value, sam)

    def set_gender_neutral (self, value, sam):
        self.set_shape_gender_neutral(value, sam)
        self.set_texture_gender_neutral(value, sam)

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

    def get_caricature (self, sam):
        return self.get_shape_caricature(sam)

    def get_caricature_neutral (self, sam):
        return self.get_shape_caricature_neutral(sam)

    def set_caricature (self, value, sam):
        self.set_shape_caricature(value, sam)
        self.set_texture_caricature(value, sam)

    def set_caricature_neutral (self, value, sam):
        self.set_shape_caricature_neutral(value, sam)
        self.set_texture_caricature_neutral(value, sam)

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

    def get_asymmetry (self, sam):
        return self.get_shape_asymmetry(sam)

    def set_asymmetry (self, value, sam):
        self.set_shape_asymmetry(value, sam)
        self.set_texture_asymmetry(value, sam)

    def get_shape_asymmetry (self, ssm):
        return np.linalg.norm(ssm.ga_data)/np.sqrt(ssm.GA, dtype=np.float32)

    def set_shape_asymmetry (self, value, ssm):
        value0 = self.get_shape_asymmetry(ssm)
        if value0 > 0:
            ssm.ga_data *= value/value0

    def get_texture_asymmetry (self, stm):
        return np.linalg.norm(stm.ta_data)/np.sqrt(stm.TA, dtype=np.float32) if stm.TA > 0 else 0.0

    def set_texture_asymmetry (self, value, stm):
        value0 = self.get_texture_asymmetry(stm)
        if value0 > 0:
            ssm.ta_data *= value/value0

    def get_race_morph (self, race, sam):
        if race not in self.current_race.morph:
            return None
        control = self.current_race.morph[race]
        shape_term = np.dot(control.gs_coeff, sam.gs_data)
        texture_term = np.dot(control.ts_coeff, sam.ts_data)
        return shape_term + texture_term + control.offset

    def set_race_morph (self, race, value, sam):
        if race not in self.current_race.morph:
            return None
        control = self.current_race.morph[race]
        value0 = self.get_race_morph(race, sam)
        sc2 = np.dot(control.gs_coeff, control.gs_coeff)
        tc2 = np.dot(control.ts_coeff, control.ts_coeff)
        k = (value - value0)/(sc2 + tc2)
        sam.gs_data += k * control.gs_coeff
        sam.ts_data += k * control.ts_coeff

    def test_model (self):
        sam = FaceGenSAM.__new__(FaceGenSAM)
        self.fix_model(sam)
        return sam

    def fix_model (self, sam):
        if isinstance(sam, FaceGenSSM):
            sam.geo_basis_version = self.geo_basis_version
            sam.GS = self.GS
            sam.GA = self.GA
            if not hasattr(sam, "gs_data") or sam.gs_data.size != sam.GS:
                sam.gs_data = np.zeros(sam.GS, dtype=np.float32)
            if not hasattr(sam, "ga_data") or sam.ga_data.size != sam.GA:
                sam.ga_data = np.zeros(sam.GA, dtype=np.float32)

        if isinstance(sam, FaceGenSTM):
            sam.tex_basis_version = self.tex_basis_version
            sam.TS = self.TS
            sam.TA = self.TA
            if not hasattr(sam, "ts_data") or sam.ts_data.size != sam.TS:
                sam.ts_data = np.zeros(sam.TS, dtype=np.float32)
            if not hasattr(sam, "ta_data") or sam.ta_data.size != sam.TA:
                sam.ta_data = np.zeros(sam.TA, dtype=np.float32)
