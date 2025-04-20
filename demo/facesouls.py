from facegen import *
import numpy as np

class FaceGenSSM: # Statistical Shape Model
    def __init__ (self, ctlfname, egmfname, trifname):
        ctlFile = FaceGenCTL(ctlfname)
        egmFile = FaceGenEGM(egmfname)
        triFile = FaceGenTRI(trifname)

        # CTL
        self.geoDim = np.uint32(ctlFile.GS)
        self.geoFaceData = np.zeros(ctlFile.GS, np.float32)
        self.geoLinCtl = np.array(ctlFile.gsLinCtl, np.float32)
        self.geoAgeCtl = np.array(ctlFile.race['All'].gsAgeCtl, np.float32)
        self.geoAgeOffset = np.float32(ctlFile.race['All'].gsAgeOffset)
        self.geoGndCtl = np.array(ctlFile.race['All'].gsGndCtl, np.float32)
        self.geoGndOffset = np.float32(ctlFile.race['All'].gsGndOffset)
        self.geoMean = np.array(ctlFile.race['All'].gsMean, np.float32)
        self.geoDensity = np.array(ctlFile.race['All'].gsDensity, np.float32).reshape((ctlFile.GS, ctlFile.GS))

        cov = np.stack((self.geoAgeCtl, self.geoGndCtl))
        cov = np.dot(cov, cov.T)
        self.covInvAG = np.linalg.inv(cov)

        uage = self.geoAgeCtl/np.linalg.norm(self.geoAgeCtl)
        ugnd = self.geoGndCtl - uage * np.dot(uage,self.geoGndCtl)
        ugnd = ugnd/np.linalg.norm(ugnd)
        uage = uage.reshape((uage.size,1))
        ugnd = ugnd.reshape((ugnd.size,1))
        self.projAG = np.dot(uage, uage.T) + np.dot(ugnd, ugnd.T)

        # EGM
        deltas = [np.array(egmFile.gsDelta[i])*egmFile.gsScale[i] for i in range(egmFile.GS)]
        self.vtxDeltas = np.permute_dims(deltas,(1,2,0))

        # TRI
        triangles1 = np.array(triFile.triangles)
        quads = np.array(triFile.quads)
        triangles2 = np.delete(quads,0,1)
        triangles3 = np.delete(quads,2,1)
        self.vtxTriangles = np.concat((triangles1, triangles2, triangles3),axis=0)
        self.vtxReference = np.array(triFile.vertices)

    def getVertices (self):
        return self.vtxReference + np.dot(self.vtxDeltas, self.geoFaceData)

    def getLinCtl (self, idx):
        return np.dot(self.geoLinCtl[idx,:], self.geoFaceData)

    def getAge (self):
        return np.dot(self.geoAgeCtl, self.geoFaceData) + self.geoAgeOffset

    def getGender (self):
        return np.dot(self.geoGndCtl, self.geoFaceData) + self.geoGndOffset

    def getCaricature (self):
        facedev = self.geoFaceData - self.geoMean
        facedevAG = self.projAG.dot(facedev)
        return np.linalg.norm(np.dot(self.geoDensity, facedev - facedevAG))/np.sqrt(self.geoDim-2, dtype = np.float32)

    def setLinCtl (self, idx, val):
        val0 = self.getLinCtl(idx)
        self.geoFaceData += (val - val0)*self.geoLinCtl[idx,:]
        return self

    def setAge (self, val):
        val0 = self.getAge()
        factors = np.dot(self.covInvAG, (val-val0, 0))
        self.geoFaceData += factors[0]*self.geoAgeCtl + factors[1]*self.geoGndCtl
        return self

    def setGender (self, val):
        val0 = self.getGender()
        factors = factors = np.dot(self.covInvAG, (0, val-val0))
        self.geoFaceData += factors[0]*self.geoAgeCtl + factors[1]*self.geoGndCtl
        return self

    def setCaricature (self, val):
        val = max(0.01, val)
        val0 = self.getCaricature()
        facedev = self.geoFaceData - self.geoMean
        facedevAG = self.projAG.dot(facedev)
        self.geoFaceData = self.geoMean + val/val0 * (facedev - facedevAG) + facedevAG
        return self

    def setZero (self):
        self.geoFaceData.fill(0.0)
        return self

class CharacterCreator (FaceGenSSM):
    def __init__ (self, ctlfname, egmfname, trifname, menufname):
        super().__init__(ctlfname, egmfname, trifname)
        self.geoSliders = dict()
        with open(menufname,'r') as f:
            s = f.read()
            rows = s.split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                assert(len(cells) > 0)
                slider = CharacterCreator.Slider(*cells)
                self.geoSliders[cells[0]] = slider
        self.update0()

    def update0 (self):
        for k,s in self.geoSliders.items():
            if s.fgID == 'A':
                s.debValue = self.getAge()
                s.guiValue = s.float2int(s.debValue)
            elif s.fgID == 'G':
                s.debValue = self.getGender()
                s.guiValue = s.float2int(s.debValue)
            elif s.fgID == 'C':
                s.debValue = self.getCaricature()
                s.guiValue = s.float2int(s.debValue)
            else:
                s.debValue = self.getLinCtl(s.fgID)
                s.guiValue = s.float2int(s.debValue)
        return self

    def update1 (self, key, value, debug = False):
        slider = self.geoSliders[key]
        if debug:
            slider.guiValue = slider.float2int(value)
            slider.debValue = value
        else:
            slider.guiValue = value
            slider.debValue = slider.int2float(value)
        if slider.fgID == 'A':
            self.setAge(slider.debValue)
        elif slider.fgID == 'G':
            self.setGender(slider.debValue)
        elif slider.fgID == 'C':
            self.setCaricature(slider.debValue)
        elif slider.menu != None:
            self.setLinCtl(slider.fgID, slider.debValue)
        else:
            self.setLinCtl(slider.fgID, 0.0)
        return self.update0()

    def update2 (self, key, value, debug = False):
        if debug:
            self.geoSliders[key].debValue = value
        else:
            self.geoSliders[key].guiValue = value
        self.setZero()
        for k,s in self.geoSliders.items():
            if not debug:
                s.debValue = s.int2float(s.guiValue)
            if s.fgID == 'A':
                self.setAge(s.debValue)
            elif s.fgID == 'G':
                self.setGender(s.debValue)
            elif s.fgID == 'C':
                self.setCaricature(s.debValue)
            elif s.menu != None:
                self.setLinCtl(s.fgID, s.debValue)
            else:
                self.setLinCtl(s.fgID, 0.0)
        return self

    class Slider:
        def __init__ (self, fgID, menu = None, label = None, rangeMin = -10.0, rangeMax = 10.0, guiMin = 0, guiMax = 255):
            try:
                self.fgID = int(fgID)
            except:
                self.fgID = fgID
            self.menu = menu
            self.label = label
            self.rangeMin = np.float32(rangeMin)
            self.rangeMax = np.float32(rangeMax)
            self.guiMin = np.uint8(guiMin)
            self.guiMax = np.uint8(guiMax)
            self.debValue = (self.rangeMin + self.rangeMax)/2.0
            self.guiValue = self.float2int(self.debValue)

        def int2float (self, x):
            aux = self.rangeMin + (self.rangeMax - self.rangeMin)*x/255
            return aux

        def float2int (self, x):
            aux = 255 * (x - self.rangeMin) / (self.rangeMax - self.rangeMin)
            aux = max(min(aux,255),0)
            return int(aux)
