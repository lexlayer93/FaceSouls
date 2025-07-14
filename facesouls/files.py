import struct
from dataclasses import dataclass

__all__ = [
    "FaceGenCTL",
    "FaceGenEGM",
    "FaceGenEGT",
    "FaceGenTRI",
    "FaceGenFG",
    "FaceGenFIM"
    ]

def struct_endian (endian):
    if endian == "little":
        return '<'
    elif endian == "big":
        return '>'
    else:
        return '='

class BinaryParser:
    def __init__ (self, buffer, *, endian=None):
        self.buffer = buffer
        self.offset = 0
        self.endian = struct_endian(endian)

    @staticmethod
    def _struct_decoder (fmt_char, process = lambda o, n: o if n != '' else o[0]):
        def decorator (func):
            def wrapper (self, n=''):
                fmt = f"{self.endian}{n}{fmt_char}"
                out = struct.unpack_from(fmt, self.buffer, self.offset)
                self.offset += struct.calcsize(fmt)
                return process(out, n)
            return wrapper
        return decorator

    @_struct_decoder('s', lambda o,_: o[0])
    def bytes_ (self, n):
        pass

    @_struct_decoder('s', lambda o,_: o[0].decode("ascii"))
    def ascii_ (self, n):
        pass

    @_struct_decoder('b')
    def char_ (self, n):
        pass

    @_struct_decoder('h')
    def short_ (self, n):
        pass

    @_struct_decoder('i')
    def int_ (self, n):
        pass

    @_struct_decoder('L')
    def ulong_ (self, n):
        pass

    @_struct_decoder('f')
    def float_ (self, n):
        pass

    @_struct_decoder('x')
    def pad_ (self, n):
        pass


class FaceGenFile:
    magic = None

    def __init__(self, src, *, endian=None):
        if isinstance(src, bytes):
            self.from_buffer(src, endian=endian)
        else:
            self.load(src, endian=endian)

    def load (self, fname, *, endian=None):
        with open(fname, 'rb') as f: b = f.read()
        self.from_buffer(b, endian=endian)

    def from_buffer (self, b, *, endian=None):
        if b[:8] != self.magic:
            raise ValueError("Wrong magic number.")
        parser = BinaryParser(b[8:], endian=endian)
        self._parse(parser)


@dataclass(init=False)
class FaceGenCTL (FaceGenFile):
    magic = b"FRCTL001"
    geo_basis_version: int
    tex_basis_version: int
    GS: int
    GA: int
    TS: int
    TA: int
    LGS: int
    lgs_controls: list["FaceGenCTL.LinCtrl"]
    LGA: int
    lga_controls: list["FaceGenCTL.LinCtrl"]
    LTS: int
    lts_controls: list["FaceGenCTL.LinCtrl"]
    LTA: int
    lta_controls: list["FaceGenCTL.LinCtrl"]
    races: dict[str, "FaceGenCTL.Race"]

    @dataclass(init=False)
    class LinCtrl:
        coeff: tuple[float,...]
        label: str
        label_len: int

    @dataclass(init=False)
    class Race:
        age_control: "FaceGenCTL.PrtOffLinCtrl"
        gnd_control: "FaceGenCTL.PrtOffLinCtrl"
        morph_controls: dict[str, "FaceGenCTL.OffLinCtrl"]
        gs_mean: tuple[float,...]
        ts_mean: tuple[float,...]
        gs_density: tuple[float,...]
        ts_density: tuple[float,...]
        combined_density: tuple[float,...]

    @dataclass(init=False)
    class PrtOffLinCtrl:
        gs_coeff: tuple[float,...]
        ts_coeff: tuple[float,...]
        gs_offset: float
        ts_offset: float

    @dataclass(init=False)
    class OffLinCtrl:
        gs_coeff: tuple[float,...]
        ts_coeff: tuple[float,...]
        offset: float

    def _parse (self, parser):
        self.geo_basis_version, self.tex_basis_version = parser.ulong_(2) # Geometry & Texture Basis Versions
        self.GS, self.GA, self.TS, self.TA = parser.ulong_(4) # N of GS, GA, TS & TA modes

        self.LGS = parser.ulong_() # N of Geometry Symmetric linear controls
        self.lgs_controls = [None] * self.LGS
        for i in range(self.LGS):
            control = FaceGenCTL.LinCtrl()
            control.coeff = parser.float_(self.GS)
            control.label_len = parser.ulong_()
            control.label = parser.ascii_(control.label_len)
            self.lgs_controls[i] = control

        self.LGA = parser.ulong_() # N of Geometry Asymmetric linear controls
        self.lga_controls = [None] * self.LGA
        for i in range(self.LGA):
            control = FaceGenCTL.LinCtrl()
            control.coeff = parser.float_(self.GA)
            control.label_len = parser.ulong_()
            control.label = parser.ascii_(control.label_len)
            self.lga_controls[i] = control

        self.LTS = parser.ulong_() # N of Texture Symmetric linear controls
        self.lts_controls = [None] * self.LTS
        for i in range(self.LTS):
            control = FaceGenCTL.LinCtrl()
            control.coeff = parser.float_(self.TS)
            control.label_len = parser.ulong_()
            control.label = parser.ascii_(control.label_len)
            self.lts_controls[i] = control

        self.LTA = parser.ulong_() # N of Texture Asymmetric linear controls
        self.lta_controls = [None] * self.LTA
        for i in range(self.LTA):
            control = FaceGenCTL.LinCtrl()
            control.coeff = parser.float_(self.TA)
            control.label_len = parser.ulong_()
            control.label = parser.ascii_(contro.label_len)
            self.lta_controls[i] = control

        self.races = dict()
        for r in ('All', 'Afro', 'Asia', 'Eind', 'Euro'):
            age_control = FaceGenCTL.PrtOffLinCtrl()
            age_control.gs_coeff = parser.float_(self.GS)
            age_control.gs_offset = parser.float_()
            age_control.ts_coeff = parser.float_(self.TS)
            age_control.ts_offset = parser.float_()
            gnd_control = FaceGenCTL.PrtOffLinCtrl()
            gnd_control.gs_coeff = parser.float_(self.GS)
            gnd_control.gs_offset = parser.float_()
            gnd_control.ts_coeff = parser.float_(self.TS)
            gnd_control.ts_offset = parser.float_()
            race = FaceGenCTL.Race()
            race.age_control = age_control
            race.gnd_control = gnd_control
            self.races[r] = race

        for r in self.races:
            self.races[r].morph_controls = dict()
            for s in self.races:
                if s == r:
                    continue
                morph_control = FaceGenCTL.OffLinCtrl()
                morph_control.gs_coeff = parser.float_(self.GS)
                morph_control.ts_coeff = parser.float_(self.TS)
                morph_control.offset = parser.float_()
                self.races[r].morph_controls[s] = morph_control

        for race in self.races.values():
            race.gs_mean = parser.float_(self.GS)
            race.ts_mean = parser.float_(self.TS)
            race.combined_density = parser.float_((self.GS+self.TS)*(self.GS+self.TS))
            race.gs_density = parser.float_(self.GS*self.GS)
            race.ts_density = parser.float_(self.TS*self.TS)


@dataclass(init=False)
class FaceGenEGM (FaceGenFile):
    magic = b"FREGM002"
    num_vertices: int
    GS: int
    GA: int
    geo_basis_version: int
    gs_scales: list[float]
    gs_deltas: list[list[tuple[int,int,int]]]
    ga_scales: list[float]
    ga_deltas: list[list[tuple[int,int,int]]]

    def _parse (self, parser):
        self.num_vertices = parser.ulong_() # Number of vertices = V + K in TRI file
        self.GS, self.GA = parser.ulong_(2) # N of GS & GA modes
        self.geo_basis_version = parser.ulong_() # Geometry Basis Version

        parser.pad_(40) # Reserved

        self.gs_scales = [None] * self.GS
        self.gs_deltas = [ [None]*self.num_vertices for _ in range(self.GS) ]
        for i in range(self.GS):
            self.gs_scales[i] = parser.float_()
            aux = parser.short_(3*self.num_vertices)
            self.gs_deltas[i] = list(zip(*(iter(aux),)*3))

        self.ga_scales = [None] * self.GA
        self.ga_deltas = [ [None]*self.num_vertices for _ in range(self.GA) ]
        for i in range(self.GA):
            self.ga_scales[i] = parser.float_()
            aux = parser.short_(3*self.num_vertices)
            self.ga_deltas[i] = list(zip(*(iter(aux),)*3))


@dataclass(init=False)
class FaceGenEGT (FaceGenFile):
    magic = b"FREGT003"
    image_height: int
    image_width: int
    TS: int
    TA: int
    tex_basis_version: int
    ts_scales: list[float]
    ts_deltas: list[list[tuple[int,int,int]]]

    def _parse (self, parser):
        self.image_height, self.image_width = parser.ulong_(2)
        self.TS, self.TA = parser.ulong_(2)
        self.tex_basis_version = parser.ulong_()

        parser.pad_(36) # Reserved

        npixels = self.image_height * self.image_width
        self.ts_scales = [None] * self.TS
        self.ts_deltas = [ [None]*npixels for _ in range(self.TS) ]
        for i in range(self.TS):
            self.ts_scales[i] = parser.float_()
            aux = parser.char_(3*npixels)
            for j in range(npixels):
                self.ts_deltas[i][j] = (aux[j], aux[j+npixels], aux[j+npixels*2])

        self.ta_scales = [None] * self.TA
        self.ta_deltas = [ [None]*npixels for _ in range(self.TA) ]
        for i in range(self.TA):
            self.ta_scales[i] = parser.float_()
            aux = parser.char_(3*npixels)
            for j in range(npixels):
                self.ta_deltas[i][j] = (aux[j], aux[j+npixels], aux[j+npixels*2])


@dataclass(init=False)
class FaceGenTRI (FaceGenFile):
    magic = b"FRTRI003"
    num_vertices: int
    num_triangles: int
    num_quads: int
    num_labelled_vertices: int
    num_labelled_surface_points: int
    num_uv: int
    extension_info: int
    num_labelled_difference_morphs: int
    num_labelled_stat_morphs: int
    num_stat_morph_vertices: int
    vertices: list[tuple[float,float,float]]
    triangles: list[tuple[int,int,int]]
    quads: list[tuple[int,int,int,int]]
    uv_vertices: list[tuple[float,float]]
    uv_triangles: list[tuple[int,int,int]]
    uv_quads: list[tuple[int,int,int,int]]

    def _parse (self, parser):

        self.num_vertices, self.num_triangles, self.num_quads = parser.int_(3) # V, T & Q
        self.num_labelled_vertices, self.num_labelled_surface_points = parser.int_(2) # LV & LS
        self.num_uv = parser.int_() # X
        self.extension_info = parser.int_()
        self.num_labelled_difference_morphs, self.num_labelled_stat_morphs = parser.int_(2) # Md & Ms
        self.num_stat_morph_vertices = parser.int_() # K

        parser.pad_(16) # Reserved

        self.vertices = [parser.float_(3) for i in range(self.num_vertices + self.num_stat_morph_vertices)]
        self.triangles = [parser.int_(3) for i in range(self.num_triangles)]
        self.quads = [parser.int_(4) for i in range(self.num_quads)]

        # UV Layout
        if self.num_uv == 0 and self.extension_info & 0x01: # per vertex
            self.uv_vertices = [parser.float_(2) for i in range(self.num_vertices)]
            self.uv_triangles = self.meshTriangles
            self.uv_quads = self.meshQuads
        elif self.num_uv > 0 and self.extension_info & 0x01: # per facet
            self.uv_vertices = [parser.float_(2) for i in range(self.num_uv)]
            self.uv_triangles = [parser.int_(3) for i in range(self.num_triangles)]
            self.uv_quads = [parser.int_(4) for i in range(self.num_quads)]

        # ETC

    def dump_obj (self, o1=0, o2=0):
        out = ""
        for v in self.vertices:
            s = ' '.join(str(x) for x in v)
            out += "v " + s + '\n'
        for vt in self.uv_vertices:
            s = ' '.join(str(uv) for uv in vt)
            out += "vt " + s + '\n'
        for t1, t2 in zip(self.triangles, self.uv_triangles):
            s = ' '.join(f"{i+o1+1}/{j+o2+1}" for i,j in zip(t1, t2))
            out += "f " + s + '\n'
        for q1, q2 in zip(self.quads, self.uv_quads):
            s = ' '.join(f"{i+o1+1}/{j+o2+1}" for i,j in zip(q1,q2))
            out += "f " + s + '\n'
        return out


@dataclass(init=False)
class FaceGenFG(FaceGenFile):
    magic = b"FRFG0001"
    geo_basis_version: int = 0
    tex_basis_version: int = 0
    GS: int = 50
    GA: int = 30
    TS: int = 50
    TA: int = 0
    detail_texture_flag: int = 0
    gs_data: list[int]
    ga_data: list[int]
    ts_data: list[int]
    ta_data: list[int]
    detail_image: bytes = b""

    def __init__ (self, src=None, *, endian=None):
        if src is not None:
            super().__init__(src, endian=endian)
        else:
            self.gs_data = [0]*self.GS
            self.ga_data = [0]*self.GA
            self.ts_data = [0]*self.TS
            self.ta_data = [0]*self.TA

    def _parse (self, parser):
        self.geo_basis_version, self.tex_basis_version = parser.ulong_(2) # Geometry & Texture Basis Versions
        self.GS, self.GA, self.TS, self.TA  = parser.ulong_(4) # N of GS, GA, TS & TA modes

        if parser.ulong_() != 0:
            raise ValueError("Must be 0")

        self.detail_texture_flag = parser.ulong_() # 0 if none, 1 if present

        self.gs_data = list(parser.short_(self.GS)) # GS coeff x 1000
        self.ga_data = list(parser.short_(self.GA)) # GA coeff x 1000
        self.ts_data = list(parser.short_(self.TS)) # TS coeff x 1000
        self.ta_data = list(parser.short_(self.TA)) # TA coeff x 1000

        if self.detail_texture_flag:
            nbytes = parser.ulong_()
            self.detail_image = parser.bytes_(nbytes)
        else:
            self.detail_image = b''

    def save (self, filename, *, endian=None):
        endian = struct_endian(endian)
        with open(filename,'wb') as f:
            f.write(self.magic)
            s = struct.pack(f'{endian}2L', self.geo_basis_version, self.tex_basis_version)
            f.write(s)
            s = struct.pack(f'{endian}4L', self.GS, self.GA, self.TS, self.TA)
            f.write(s)
            s = struct.pack(f'{endian}2L', 0, self.detail_texture_flag)
            f.write(s)
            s = struct.pack(f'{endian}{self.GS}h', *(self.gs_data))
            f.write(s)
            s = struct.pack(f'{endian}{self.GA}h', *(self.ga_data))
            f.write(s)
            s = struct.pack(f'{endian}{self.TS}h', *(self.ts_data))
            f.write(s)
            s = struct.pack(f'{endian}{self.TA}h', *(self.ta_data))
            f.write(s)
            s = struct.pack(f'{endian}L', len(self.detail_image))
            f.write(s)
            f.write(self.detail_image)


@dataclass(init=False)
class FaceGenFIM (FaceGenFile):
    magic = b"FIMFF001"
    image_width: int
    image_height: int
    image_uv: list[tuple[float,float]]

    def _parse (self, parser):
        self.image_width, self.image_height = parser.ulong_(2)

        parser.pad_(48) # Unused

        aux = parser.float_(2*self.image_width*self.image_height)
        self.image_uv = list(zip(*(iter(aux))*2)) # image resampling coordinates
