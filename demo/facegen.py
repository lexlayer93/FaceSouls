import struct

def Decoder (b, endian = '<'):
    def bread (fmt):
        fmt = endian + fmt
        out = struct.unpack_from(fmt, b, bread.offset)
        bread.offset += struct.calcsize(fmt)
        return out
    bread.offset = 0
    return bread

class FaceGenCTL:
    magic = "FRCTL001"

    def __init__(self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename, 'rb')
        b = f.read()
        f.close()

        ctl_read = Decoder(b)
        assert(ctl_read('8s')[0].decode() == FaceGenCTL.magic)

        self.geoBasisVersion, self.texBasisVersion = ctl_read('2L') # Geometry & Texture Basis Versions
        self.GS, self.GA, self.TS, self.TA = ctl_read('4L') # N of GS, GA, TS & TA modes

        self.LGS, = ctl_read('L') # N of Geometry Symmetric linear controls
        self.gsLinCtl = [None] * self.LGS
        self.gsLinLabel = [None] * self.LGS
        for i in range(self.LGS):
            self.gsLinCtl[i] = ctl_read(f'{self.GS}f')
            lablen, = ctl_read('L')
            self.gsLinLabel[i] = ctl_read(f'{lablen}s')[0].decode()

        self.LGA, = ctl_read('L') # N of Geometry Asymmetric linear controls
        self.gaLinCtl = [None] * self.LGA
        self.gaLinLabel = [None] * self.LGA
        for i in range(self.LGA):
            self.gaLinCtl[i] = ctl_read(f'{self.GA}f')
            lablen, = ctl_read('L')
            self.gaLinLabel[i] = ctl_read(f'{lablen}s')[0].decode()

        self.LTS, = ctl_read('L') # N of Texture Symmetric linear controls
        self.tsLinCtl = [None] * self.LTS
        self.tsLinLabel = [None] * self.LTS
        for i in range(self.LTS):
            self.tsLinCtl[i] = ctl_read(f'{self.TS}f')
            lablen, = ctl_read('L')
            self.tsLinLabel[i] = ctl_read(f'{lablen}s')[0].decode()

        self.LTA, = ctl_read('L') # N of Texture Asymmetric linear controls
        self.taLinCtl = [None] * self.LTA
        self.taLinLabel = [None] * self.LTA
        for i in range(self.LTA):
            self.taLinCtl[i] = ctl_read(f'{self.TA}f')
            lablen, = ctl_read('L')
            self.taLinLabel[i] = ctl_read(f'{lablen}s')[0].decode()

        class ctlRace:
            pass

        self.race = {'All': ctlRace(), 'Afro': ctlRace(), 'Asia': ctlRace(), 'Eind': ctlRace(), 'Euro': ctlRace()}

        for r in self.race:
            self.race[r].gsAgeCtl = ctl_read(f'{self.GS}f')
            self.race[r].gsAgeOffset, = ctl_read('f')
            self.race[r].tsAgeCtl = ctl_read(f'{self.TS}f')
            self.race[r].tsAgeOffset, = ctl_read('f')
            self.race[r].gsGndCtl = ctl_read(f'{self.GS}f')
            self.race[r].gsGndOffset, = ctl_read('f')
            self.race[r].tsGndCtl = ctl_read(f'{self.TS}f')
            self.race[r].tsGndOffset, = ctl_read('f')

        for r in self.race:
            self.race[r].gtsMorph = dict()
            for s in self.race:
                if s != r:
                    self.race[r].gtsMorph[s] = ctl_read(f'{self.GS+self.TS+1}f')

        for r in self.race:
            self.race[r].gsMean = ctl_read(f'{self.GS}f')
            self.race[r].tsMean = ctl_read(f'{self.TS}f')
            self.race[r].gtsCombinedDensity = ctl_read(f'{(self.GS+self.TS)*(self.GS+self.TS)}f')
            self.race[r].gsDensity = ctl_read(f'{self.GS*self.GS}f')
            self.race[r].tsDensity = ctl_read(f'{self.TS*self.TS}f')


class FaceGenEGM:
    magic = "FREGM002"

    def __init__ (self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        egm_unpack = Decoder(b)
        assert(egm_unpack('8s')[0].decode() == FaceGenEGM.magic)

        self.nVert, = egm_unpack('L') # Number of vertices
        self.GS, self.GA = egm_unpack('2L') # N of SS & AS modes
        self.geoBasisVersion, = egm_unpack('L') # Geometry Basis Version

        egm_unpack('40s') # Reserved

        self.gsScale = [None] * self.GS
        self.gsDelta = [ [None]*self.nVert for _ in range(self.GS) ]

        for i in range(self.GS):
            self.gsScale[i], = egm_unpack('f')
            for j in range(self.nVert):
                self.gsDelta[i][j] = egm_unpack('3h')


        self.gaScale = [None] * self.GA
        self.gaDelta = [ [None]*self.nVert for _ in range(self.GA) ]
        for i in range(self.GA):
            self.gaScale[i], = egm_unpack('f')
            for j in range(self.nVert):
                self.gaDelta[i][j] = egm_unpack('3h')


class FaceGenTRI:
    magic = "FRTRI003"

    def __init__ (self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        tri_unpack = Decoder(b)
        assert(tri_unpack('8s')[0].decode() == FaceGenTRI.magic)

        self.V, self.T, self.Q = tri_unpack('3i') # N of vertices, triangles & quads
        self.LV, self.LS = tri_unpack('2i') # N of labelled vertices & surface points
        self.X, = tri_unpack('i') # N of texture coordinates
        self.ext, = tri_unpack('i') # Extension info
        self.Md, self.Ms = tri_unpack('2i') # N of labelled difference & stat morphs
        self.K, = tri_unpack('i') # total N of stat morph vertices
        tri_unpack('16c') # Reserved

        self.vertices = [tri_unpack('3f') for i in range(self.V + self.K)]
        self.triangles = [tri_unpack('3i') for i in range(self.T)]
        self.quads = [tri_unpack('4i') for i in range(self.Q)]

        # ETC


class FaceGenFG:
    magic = "FRFG0001"
    def __init__(self, filename = None):
        if filename is None:
            self.geoBasisVersion = 2001060901
            self.texBasisVersion = 81
            self.GS = 50
            self.GA = 30
            self.TS = 50
            self.TA = 0
            self.detailTextureFlag = 0
            self.gsData = [0]*self.GS
            self.saData = [0]*self.GA
            self.tsData = [0]*self.TS
            self.taData = [0]*self.TA
        else:
            self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        fg_read = Decoder(b)
        assert(fg_read('8s')[0].decode() == FaceGenFG.magic)

        self.geoBasisVersion, self.texBasisVersion = fg_read('2L') # Geometry & Texture Basis Versions
        self.GS, self.GA, self.TS, self.TA  = fg_read('4L') # N of GS, GA, TS & TA modes

        assert(fg_read('L') == 0) # Must be 0
        self.detailTextureFlag, = fg_read('L')

        self.gsData = list(fg_read(f'{self.GS}h')) # GS coeff x 1000
        self.gaData = list(fg_read(f'{self.GA}h')) # GA coeff x 1000
        self.tsData = list(fg_read(f'{self.TS}h')) # TS coeff x 1000
        self.taData = list(fg_read(f'{self.TA}h')) # TA coeff x 1000
        # ETC

    def save (self, filename):
        with open(filename,'wb') as f:
            f.write(self.magic)
            s = struct.pack('<8L', self.geoBasisVersion, self.texBasisVersion, self.GS, self.GA, self.TS, self.TA, 0, self.detailTextureFlag)
            f.write(s)
            s = struct.pack(f'<{self.GS}h', *(self.gsCoef))
            f.write(s)
            s = struct.pack(f'<{self.GA}h', *(self.gaCoef))
            f.write(s)
            s = struct.pack(f'<{self.TS}h', *(self.tsCoef))
            f.write(s)
            s = struct.pack(f'<{self.TA}h', *(self.taCoef))
            f.write(s)
