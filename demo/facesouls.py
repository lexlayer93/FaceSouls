from facegen import *
import numpy as np

class FaceGenSSM: # Statistical Shape Model
    def __init__ (self, lof):
        if len(lof) != 3:
            raise AssertionError("Wrong number of files.")

        ctlFile = FaceGenCTL(lof[0])
        egmFile = FaceGenEGM(lof[1])
        triFile = FaceGenTRI(lof[2])

        # CTL
        self.GS = np.uint32(ctlFile.GS)
        self.GA = np.uint32(ctlFile.GA)
        self.gsFaceData = np.zeros(self.GS, np.float32)
        self.gaFaceData = np.zeros(self.GA, np.float32)
        self.gsLinCtl = np.array(ctlFile.gsLinCtl, np.float32)
        self.gaLinCtl = np.array(ctlFile.gaLinCtl, np.float32)
        self.gsAgeCtl = np.array(ctlFile.race['All'].gsAgeCtl, np.float32)
        self.gsAgeOffset = np.float32(ctlFile.race['All'].gsAgeOffset)
        self.gsGndCtl = np.array(ctlFile.race['All'].gsGndCtl, np.float32)
        self.gsGndOffset = np.float32(ctlFile.race['All'].gsGndOffset)
        self.gsMean = np.array(ctlFile.race['All'].gsMean, np.float32)
        self.gsDensity = np.array(ctlFile.race['All'].gsDensity, np.float32).reshape((self.GS, self.GS))

        cov = np.stack((self.gsAgeCtl, self.gsGndCtl))
        cov = np.dot(cov, cov.T)
        self._gsCovInvAG = np.linalg.inv(cov)

        uage = self.gsAgeCtl/np.linalg.norm(self.gsAgeCtl)
        ugnd = self.gsGndCtl - uage * np.dot(uage,self.gsGndCtl)
        ugnd = ugnd/np.linalg.norm(ugnd)
        uage = uage.reshape((uage.size,1))
        ugnd = ugnd.reshape((ugnd.size,1))
        self._gsProjAG = np.dot(uage, uage.T) + np.dot(ugnd, ugnd.T)

        # EGM
        deltas = [np.array(egmFile.gsDelta[i])*egmFile.gsScale[i] for i in range(egmFile.GS)]
        self.gsDeltas = np.permute_dims(deltas,(1,2,0)).astype(np.float32)
        deltas = [np.array(egmFile.gaDelta[i])*egmFile.gaScale[i] for i in range(egmFile.GA)]
        self.gaDeltas = np.permute_dims(deltas,(1,2,0)).astype(np.float32)

        # TRI
        triangles1 = np.array(triFile.meshTriangles)
        quads = np.array(triFile.meshQuads)
        triangles2 = np.delete(quads,0,1)
        triangles3 = np.delete(quads,2,1)
        self.geoTriangles = np.concat((triangles1, triangles2, triangles3),axis=0)
        self.geoVertexRef = np.array(triFile.meshVertices, np.float32)

    def getVertices (self):
        return self.geoVertexRef + np.dot(self.gsDeltas, self.gsFaceData) + np.dot(self.gaDeltas, self.gaFaceData)

    def getSymCtl (self, idx):
        return np.dot(self.gsLinCtl[idx,:], self.gsFaceData)

    def getAge (self):
        return np.dot(self.gsAgeCtl, self.gsFaceData) + self.gsAgeOffset

    def getGender (self):
        return np.dot(self.gsGndCtl, self.gsFaceData) + self.gsGndOffset

    def getCaricature (self):
        facedev = self.gsFaceData - self.gsMean
        facedevAG = self._gsProjAG.dot(facedev)
        return np.linalg.norm(np.dot(self.gsDensity, facedev - facedevAG))/np.sqrt(self.GS-2, dtype = np.float32)

    def getAsymCtl (self, idx):
        return np.dot(self.gaLinCtl[idx,:], self.gaFaceData)

    def getAsymmetry (self):
        return np.linalg.norm(self.gaFaceData)/np.sqrt(self.GA, dtype = np.float32)

    def setSymCtl (self, idx, val):
        val0 = self.getSymCtl(idx)
        self.gsFaceData += (val - val0) * self.gsLinCtl[idx,:]
        return self

    def setAge (self, val):
        val0 = self.getAge()
        factors = np.dot(self._gsCovInvAG, (val-val0, 0))
        self.gsFaceData += factors[0]*self.gsAgeCtl + factors[1]*self.gsGndCtl
        return self

    def setGender (self, val):
        val0 = self.getGender()
        factors = np.dot(self._gsCovInvAG, (0, val-val0))
        self.gsFaceData += factors[0]*self.gsAgeCtl + factors[1]*self.gsGndCtl
        return self

    def setCaricature (self, val):
        val = max(0.01, val)
        val0 = self.getCaricature()
        facedev = self.gsFaceData - self.gsMean
        facedevAG = self._gsProjAG.dot(facedev)
        self.gsFaceData = self.gsMean + val/val0 * (facedev - facedevAG) + facedevAG
        return self

    def setAsymCtl (self, idx, val):
        val0 = self.getAsymCtl(idx)
        self.gaFaceData += (val - val0) * self.gaLinCtl[idx,:]
        return self

    def setAsymmetry (self, val):
        val0 = self.getAsymmetry()
        if val0 != 0:
            self.gaFaceData *= val/val0
        return self

    def setZero (self):
        self.gsFaceData.fill(0.0)
        self.gaFaceData.fill(0.0)
        return self


class CharacterCreator (FaceGenSSM):
    def __init__ (self, lof):
        super().__init__(lof[:3])
        self.geoSliders = dict()
        with open(lof[3],'r') as f:
            s = f.read()
            rows = s.split(';')
            del rows[-1]
            for r in rows:
                cells = list(map(lambda s: s.strip(), r.split(',')))
                if len(cells) > 0:
                    slider = CharacterCreator.Slider(*cells)
                    self.geoSliders[cells[0]] = slider
        self.updateSliders()

    def updateSliders (self):
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
                s.debValue = self.getSymCtl(s.fgID)
                s.guiValue = s.float2int(s.debValue)
        return self

    def changeFace1 (self, key, value, debug = False): # Des & DS1
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
            self.setSymCtl(slider.fgID, slider.debValue)
        else:
            self.setSymCtl(slider.fgID, 0.0)
        return self.updateSliders()

    def changeFace2 (self, key, value, debug = False): # BB, DS3 & ER
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
                self.setSymCtl(s.fgID, s.debValue)
            else:
                self.setSymCtl(s.fgID, 0.0)
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
            y = self.rangeMin + (self.rangeMax - self.rangeMin) * x/255
            return y

        def float2int (self, x):
            y = 255 * (x - self.rangeMin) / (self.rangeMax - self.rangeMin)
            y = max(min(y,255),0)
            return int(y)
