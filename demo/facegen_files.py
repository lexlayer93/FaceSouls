import struct

def _Decoder (b, start = 0, endian = '<'):
    def bread (fmt):
        fmt = endian + fmt
        out = struct.unpack_from(fmt, b, bread.offset)
        bread.offset += struct.calcsize(fmt)
        return out
    bread.offset = start
    return bread


class FaceGenCTL:
    magic = b"FRCTL001"

    def __init__(self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename, 'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenCTL.magic:
            raise AssertionError("Wrong magic number.")

        ctl_read = _Decoder(b, 8)

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
    magic = b"FREGM002"

    def __init__ (self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenEGM.magic:
            raise AssertionError("Wrong magic number.")

        egm_read = _Decoder(b, 8)

        self.nVert, = egm_read('L') # Number of vertices = V + K in TRI file
        self.GS, self.GA = egm_read('2L') # N of GS & GA modes
        self.geoBasisVersion, = egm_read('L') # Geometry Basis Version

        egm_read('40s') # Reserved

        self.gsScale = [None] * self.GS
        self.gsDelta = [ [None]*self.nVert for _ in range(self.GS) ]
        for i in range(self.GS):
            self.gsScale[i], = egm_read('f')
            for j in range(self.nVert):
                self.gsDelta[i][j] = egm_read('3h')

        self.gaScale = [None] * self.GA
        self.gaDelta = [ [None]*self.nVert for _ in range(self.GA) ]
        for i in range(self.GA):
            self.gaScale[i], = egm_read('f')
            for j in range(self.nVert):
                self.gaDelta[i][j] = egm_read('3h')


class FaceGenEGT:
    magic = b"FREGT003"

    def __init__(self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename, 'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenEGT.magic:
            raise AssertionError("Wrong magic number.")

        egt_read = _Decoder(b, 8)

        self.nRow, self.nCol = egt_read('2L') # of image
        self.TS, self.TA = egt_read('2L')
        self.texBasisVersion, = egt_read('L')
        egt_read('36s') # Reserved

        self.tsScale = [None] * self.TS
        self.tsRDelta = [None] * self.TS
        self.tsGDelta = [None] * self.TS
        self.tsBDelta = [None] * self.TS
        for i in range(self.TS):
            self.tsScale[i] = egt_read('f')
            self.tsRDelta[i] = egt_read(f'{self.nRow*self.nCol}b')
            self.tsGDelta[i] = egt_read(f'{self.nRow*self.nCol}b')
            self.tsBDelta[i] = egt_read(f'{self.nRow*self.nCol}b')

        self.taScale = [None] * self.TA
        self.taRDelta = [None] * self.TA
        self.taGDelta = [None] * self.TA
        self.taBDelta = [None] * self.TA
        for i in range(self.TA):
            self.taScale[i] = egt_read('f')
            self.taRDelta[i] = egt_read(f'{self.nRow*self.nCol}b')
            self.taGDelta[i] = egt_read(f'{self.nRow*self.nCol}b')
            self.taBDelta[i] = egt_read(f'{self.nRow*self.nCol}b')


class FaceGenFIM:
    magic = b"FIMFF001"

    def __init__ (self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename, 'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenFIM.magic:
            raise AssertionError("Wrong magic number.")

        fim_read = _Decoder(b, 8)

        self.imageWidth = fim_read('L')
        self.imageHeight = fim_read('L')
        fim_read('48s') # Unused

        aux = fim_read(f'{2*self.imageWidth*self.imageHeight}f')
        self.imageUV = list(zip(*[iter(aux)]*2)) # image resampling coordinates


class FaceGenTRI:
    magic = b"FRTRI003"

    def __init__ (self, filename):
        self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenTRI.magic:
            raise AssertionError("Wrong magic number.")

        tri_read = _Decoder(b, 8)

        self.nVert, self.nTri, self.nQuad = tri_read('3i') # N of vertices (V), triangles (T) & quads (Q)
        self.nLVert, self.nLSurf = tri_read('2i') # N of labelled vertices (LV) & surface points (LS)
        self.nUV, = tri_read('i') # N of texture coordinates (X)
        self.extInfo, = tri_read('i') # Extension info
        self.nLDiffMorph, self.nLStatMorph = tri_read('2i') # N of labelled difference (Md) & stat morphs (Ms)
        self.nSMVert, = tri_read('i') # total N of stat morph vertices (K)
        tri_read('16s') # Reserved

        self.meshVertices = [tri_read('3f') for i in range(self.nVert + self.nSMVert)]
        self.meshTriangles = [tri_read('3i') for i in range(self.nTri)]
        self.meshQuads = [tri_read('4i') for i in range(self.nQuad)]

        # UV Layout
        if self.nUV == 0 and self.extInfo & 0x01:
            self.uvVertices = [tri_read('2f') for i in range(self.nVert)]
            self.uvTriangles = self.meshTriangles
            self.uvQuads = self.meshQuads

        elif self.nUV > 0 and self.extInfo & 0x01:
            self.uvVertices = [tri_read('2f') for i in range(self.nUV)]
            self.uvTriangles = [tri_read('3i') for i in range(self.nTri)]
            self.uvQuads = [tri_read('4i') for i in range(self.nQuad)]

        # ETC


class FaceGenFG:
    magic = b"FRFG0001"
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
            self.detailImage = b''
        else:
            self.load(filename)

    def load (self, filename):
        f = open(filename,'rb')
        b = f.read()
        f.close()

        if b[:8] != FaceGenFG.magic:
            raise AssertionError("Wrong magic number.")

        fg_read = _Decoder(b, 8)

        self.geoBasisVersion, self.texBasisVersion = fg_read('2L') # Geometry & Texture Basis Versions
        self.GS, self.GA, self.TS, self.TA  = fg_read('4L') # N of GS, GA, TS & TA modes

        if fg_read('L') != 0:
            raise ValueError("Must be 0")

        self.detailTextureFlag, = fg_read('L') # 0 if none, 1 if present

        self.gsData = list(fg_read(f'{self.GS}h')) # GS coeff x 1000
        self.gaData = list(fg_read(f'{self.GA}h')) # GA coeff x 1000
        self.tsData = list(fg_read(f'{self.TS}h')) # TS coeff x 1000
        self.taData = list(fg_read(f'{self.TA}h')) # TA coeff x 1000

        if self.detailTextureFlag:
            nbytes = fg_read('L')
            self.detailImage, = fg_read(f'{nbytes}s')
        else:
            self.detailImage = b''

    def save (self, filename):
        with open(filename,'wb') as f:
            f.write(self.magic)
            s = struct.pack('<8L', self.geoBasisVersion, self.texBasisVersion, self.GS, self.GA, self.TS, self.TA, 0, self.detailTextureFlag)
            f.write(s)
            s = struct.pack(f'<{self.GS}h', *(self.gsData))
            f.write(s)
            s = struct.pack(f'<{self.GA}h', *(self.gaData))
            f.write(s)
            s = struct.pack(f'<{self.TS}h', *(self.tsData))
            f.write(s)
            s = struct.pack(f'<{self.TA}h', *(self.taData))
            f.write(s)
            s = struct.pack('<L', len(self.detailImage))
            f.write(s)
            f.write(self.detailImage)
