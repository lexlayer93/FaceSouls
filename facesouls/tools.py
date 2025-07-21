import os.path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import dlib

__all__ = [
    "faceplot",
    "find_landmarks",
    "facemesh_to_fg",
    "facegen_to_cc"
    ]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLIB_PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_DETECTOR = None
DLIB_PREDICTOR = None


def faceplot (ax, vertices, triangles, *,
              persp="persp", rotation=0, light_alt=25,
              **kwargs):
    kwargs.setdefault("color", "peachpuff")
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("linewidth", 0)
    kwargs.setdefault("shade", True)
    kwargs.setdefault("antialiased", False)
    kwargs.setdefault("lightsource", LightSource(azdeg=0, altdeg=light_alt))

    ax.view_init(vertical_axis='y', elev=0, azim=rotation, roll=0)
    polyc = ax.plot_trisurf(vertices[:,0], vertices[:,1], triangles, vertices[:,2], **kwargs)
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
    ax.set_facecolor('k')
    ax.set_axis_off()
    ax.set_proj_type(persp)
    return polyc


def find_landmarks (mesh, **kwargs):
    global DLIB_DETECTOR, DLIB_PREDICTOR
    if DLIB_DETECTOR is None:
        DLIB_DETECTOR = dlib.get_frontal_face_detector()
    if DLIB_PREDICTOR is None:
        DLIB_PREDICTOR = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

    vertices = mesh.vertices
    triangles = mesh.triangles

    fig = plt.figure(facecolor='k', dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    faceplot(ax, vertices, triangles, persp="ortho", **kwargs)
    fig.canvas.draw()
    buf, (width, height) = fig.canvas.print_to_buffer()
    plt.close(fig)

    img_rgba = np.frombuffer(buf, np.uint8).reshape((height,width,4))
    r = img_rgba[:,:,0].astype(np.float32)
    g = img_rgba[:,:,1].astype(np.float32)
    b = img_rgba[:,:,2].astype(np.float32)
    gray = 0.114*b + 0.587*g + 0.299*r
    img_gray = gray.round().astype(np.uint8)
    mask = img_gray > 0
    coords = np.argwhere(mask)

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    img_gray = img_gray[y1:y2, x1:x2]
    height, width = img_gray.shape[:2]

    img_gray = np.ascontiguousarray(img_gray)
    face = DLIB_DETECTOR(img_gray)[0]
    shape = DLIB_PREDICTOR(img_gray, face)
    lm_pix = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    mesh_xy = vertices[:,:2]
    x_min, y_min = mesh_xy.min(axis=0)
    x_max, y_max = mesh_xy.max(axis=0)

    lm_xy = np.empty_like(lm_pix, dtype=float)
    lm_xy[:,0] = (lm_pix[:,0]+0.5)*(x_max - x_min)/width + x_min
    lm_xy[:,1] = (lm_pix[:,1]+0.5)*(y_min - y_max)/height + y_max

    diffs = lm_xy[:, np.newaxis, :] - mesh_xy
    sq_distances = np.sum(diffs**2, axis=2)
    landmarks = np.argmin(sq_distances, axis=1)

    return landmarks


def facemesh_to_fg (
    mesh, ssm, *,
    indices=None, landmarks=None,
    minimize="error", asymmetry=True, iterations=1,
    wz=0.0, wl=1.0
    ):
    targets = mesh.vertices

    if asymmetry:
        deltas = np.concatenate([ssm.gs_deltas, ssm.ga_deltas], axis=2)
    else:
        deltas = np.copy(ssm.gs_deltas)

    verts0 = ssm.vertices
    verts = np.copy(verts0)

    nv, ndim = verts0.shape

    if indices is None: indices = np.arange(nv)
    verts[indices] = targets

    weights = np.zeros((nv,ndim), dtype=np.float32)
    weights[indices,:] = 1.0

    targets_z = targets[:,2] - verts[:,2].min()
    targets_z = np.clip(targets_z, 0.0, None)
    targets_z /= targets_z.max()
    if wz < 0:
        targets_z = 1 - targets_z
        wz = -wz
    weights[indices,:] *= np.power(targets_z[:,None], wz)

    if landmarks is not None:
        weights[landmarks,:2] *= wl

    deltas *= weights[:,:,None]
    verts *= weights
    verts0 *= weights

    ns = deltas.shape[2]
    D = deltas.reshape(-1, ns)
    Dinv = np.linalg.pinv(D)

    if iterations > 0:
        if minimize == "error":
            N = np.eye(3*nv) - (D @ Dinv)
        elif minimize == "effort":
            N = Dinv.T @ Dinv
        else:
            N = np.eye(3*nv)
        Sx = np.array([[0, 0, 0],[0,0,-1],[0,1,0]], dtype = np.float32)
        Sy = np.array([[0, 0, 1],[0,0,0],[-1,0,0]], dtype = np.float32)
        Sz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype = np.float32)
        exyz = (weights[:,:,None] * np.eye(3)).reshape(-1,3)

    xt, yt, zt, at, bt, ct, kt = np.zeros(7, dtype=np.float32)
    for _ in range(iterations):
        Sx_v = Sx.dot(verts.T).flatten('F').reshape((3*nv,1))
        Sy_v = Sy.dot(verts.T).flatten('F').reshape((3*nv,1))
        Sz_v = Sz.dot(verts.T).flatten('F').reshape((3*nv,1))
        I_v = verts.flatten('C').reshape((3*nv,1))

        blk = np.concatenate([exyz, Sx_v, Sy_v, Sz_v, I_v], axis=1)
        H = blk.T @ N @ blk
        J = blk.T @ N @ (verts - verts0).flatten()
        x, y, z, a, b, c, k = tuple(np.linalg.inv(H).dot(-J).reshape(7))

        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]], dtype=np.float32)
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]], dtype=np.float32)
        Rx = np.array([[1, 0, 0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        verts = (1+k)*verts.dot(R.T) + np.multiply(weights,[x,y,z])

    fit = Dinv.dot((verts - verts0).flatten())

    weighted_error = np.sum((verts0 + np.dot(deltas,fit) - verts)**2)/np.sum(weights**2)

    best = ssm.copy()
    best.gs_data += fit[:ssm.GS]
    if asymmetry:
        best.ga_data += fit[ssm.GS:]
    else:
        best.ga_data.fill(0.0)

    return (best,
            weighted_error)


def facegen_to_cc (
    target, character_creator, *,
    mode=0, maxiter=100, **kwargs
    ):
    cc = character_creator
    sequence = cc.sequence
    available = [key for tab in cc.menu.values() for key in tab]

    # available should be <= sequence
    if not set(available).issubset(set(sequence)):
        available = [key for key in sequence if not cc.sliders[key].debug_only]

    lgs_seq = [key for key in sequence if key//100 == 1]
    lgs_avail = [key for key in available if key//100 == 1]
    lgs_idx = [key % 100 for key in lgs_avail]

    # cumulative effect of shape sliders
    mtx = np.zeros_like(cc.lgs_coeffs.T)
    I = np.eye(character_creator.GS)
    Nt = np.copy(I)
    for key in lgs_seq[::-1]:
        idx = key % 100
        c = cc.lgs_coeffs[idx,:].reshape(-1,1)
        N = I - np.dot(c,c.T)
        mtx[:,idx] = Nt.dot(c).reshape(-1)
        Nt = Nt.dot(N)
    mfit = mtx[:,lgs_idx]

    # initial state before shape sliders
    replica = cc.models[0].copy()

    if cc.all_at_once:
        preseq = {k:cc.sliders[k].value for k in sequence if k < 100}
        cc.set_zero(replica)
        for key,value in preseq.items():
            cc.set_control(key, value, replica)
    else:
        preseq = dict()

    s0 = Nt.dot(replica.gs_data)
    st = target.gs_data
    smin, smax = cc.shape_sym_range


    # faster sequence, with clip
    def apply_seq (p):
        return np.clip(s0 + mfit.dot(p), smin, smax)

    def in_range (p):
        s = apply_seq(p)
        onezero = np.where((s>=smin) & (s <=smax), 1.0, 0.0)
        return np.diag(onezero)

    if mode <= 0: # minimize shape difference
        def residual (p):
            s = apply_seq(p)
            return s - st
        def jacobian (p):
            return in_range(p).dot(mfit)

    elif mode == 1: # minimize features difference
        def residual (p):
            s = apply_seq(p)
            return np.dot(cc.lgs_coeffs, s - st)
        def jacobian (p):
            return cc.lgs_coeffs.dot(in_range(p)).dot(mfit)

    elif mode >= 2: # minimize vertex difference
        deltas = replica.gs_deltas.reshape(-1, replica.gs_data.size)
        def residual (p):
            s = apply_seq(p)
            return np.dot(deltas, s - st)
        def jacobian (p):
            return deltas.dot(in_range(p)).dot(mfit)

    # sliders bounds
    ranges = np.array([cc.sliders[key].available_range for key in lgs_avail],
                            dtype=np.float32)
    ranges.sort(axis=1)

    # initial guess
    p0 = np.linalg.pinv(mfit).dot(st-s0)
    if (np.any(p0 < ranges[:,0]) or
        np.any(p0 > ranges[:,1])):
        p0 = np.zeros_like(p0, dtype=np.float32)

    # optimization
    kwargs.setdefault("max_nfev", maxiter)
    result = optimize.least_squares(residual,
                                    p0,
                                    jac=jacobian,
                                    bounds=(ranges[:,0],ranges[:,1]),
                                    **kwargs)

    solution = dict(zip(lgs_avail, result.x))
    replica.gs_data = apply_seq(result.x)

    solution = {**preseq, **solution}

    result.nit = result.nfev
    return (solution, replica, result)
