import numpy as np
from trimesh import load
from trimesh.base import Trimesh
from trimesh.registration import procrustes, icp, nricp_amberg, nricp_sumner

__all__ = [
    "Facemesh"
    ]

class Facemesh:
    def __init__ (self, vertices, triangles, *args,
                  landmarks=[], **kwargs):
        self.__trimesh = Trimesh(vertices, triangles, *args,
                                 process=False, **kwargs)
        self.landmarks = np.asarray(landmarks, dtype=int).copy()

    @classmethod
    def fromfile (cls, src, *args, **kwargs):
        new = object.__new__(cls)
        new.__trimesh = load(src, *args, force="mesh", **kwargs)
        new.landmarks = np.empty(0, dtype=int)
        return new

    @classmethod
    def fromfg (cls, src):
        return cls(src.vertices, src.triangles)

    @property
    def vertices (self):
        return self.__trimesh.vertices

    @vertices.setter
    def vertices (self, verts):
        self.__trimesh.vertices = verts

    @property
    def triangles (self):
        return self.__trimesh.faces

    @triangles.setter
    def triangles (self, tris):
        self.__trimesh.faces = tris

    @property
    def bounds (self):
        return self.__trimesh.bounds

    @property
    def keypoints (self):
        return self.__trimesh.vertices[self.landmarks]

    def distance (self, to):
        _, dist, _ = to.__trimesh.nearest.on_surface(self.vertices)
        return np.median(dist)

    def nearest_vertex (self, points):
        _, indices = self.__trimesh.nearest.vertex(points=points)
        return indices

    def sliced (self, k=0.0):
        x = self.vertices[:,0]
        z = self.vertices[:,2]
        imin, imax = x.argmin(), x.argmax()
        z0 = (z[imin] + z[imax])/2.0
        depth = self.bounds[1,2] - z0
        zs = z0 + k*depth

        sliced = self.__trimesh.slice_plane((0,0,zs), (0,0,1))
        if self.landmarks.size > 0:
            mask = self.keypoints[:,2] >= zs
            _, newlm = sliced.nearest.vertex(points=self.keypoints[mask])
            return Facemesh(sliced.vertices, sliced.faces, landmarks=newlm)
        else:
            return Facemesh(sliced.vertices, sliced.faces)

    def cropped (self, xtol=0.05, ytol=0.1, ztol=0.05):
        if self.landmarks.size == 0:
            return Facemesh(self.vertices, self.triangles)

        lm_verts = self.keypoints
        xmin, ymin, zmin = lm_verts.min(axis=0)
        xmax, ymax, zmax = lm_verts.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin
        depth = zmax - zmin

        cropped = (self.__trimesh
                .slice_plane((0,ymin - ytol*height,0), (0,1,0))
                .slice_plane((0,ymax + ytol*height,0), (0,-1,0))
                .slice_plane((0,0,zmin - ztol*depth), (0,0,1))
                .slice_plane((0,0,zmax + ztol*depth), (0,0,-1))
                .slice_plane((xmin - xtol*width,0,0), (1,0,0))
                .slice_plane((xmax + xtol*width,0,0), (-1,0,0))
                )

        _, newlm = cropped.nearest.vertex(points=lm_verts)
        return Facemesh(cropped.vertices, cropped.faces, landmarks=newlm)

    def aligned (self, target):
        matrix0, _, _ = procrustes(self.keypoints,
                                   target.keypoints,
                                   reflection=False)

        _, transformed, _ = icp(self.vertices,
                                target.vertices,
                                initial = matrix0,
                                reflection=False)

        return Facemesh(transformed, self.triangles, landmarks=self.landmarks)

    def fitted (self, target, *,
                force=False, **kwargs):
        kwargs.setdefault("distance_threshold", 1e-2)

        transformed = nricp_amberg(self.__trimesh, target.__trimesh,
                                   source_landmarks = self.landmarks,
                                   target_positions = target.keypoints,
                                   **kwargs)

        if force:
            dt = kwargs["distance_threshold"]
            closest, distance, _ = target.__trimesh.nearest.on_surface(points=transformed)
            mask = distance <= dt
            transformed[mask] = closest[mask]

        return Facemesh(transformed, self.triangles, landmarks=self.landmarks)
