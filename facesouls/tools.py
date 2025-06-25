import os.path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import dlib
from trimesh.base import Trimesh
from trimesh.proximity import closest_point
from trimesh.registration import procrustes, icp, nricp_amberg, nricp_sumner
from trimesh.triangles import points_to_barycentric, barycentric_to_points
from trimesh.transformations import compose_matrix, decompose_matrix


__all__ = [
    "facemesh_plot",
    "facemesh_landmarks",
    "facemesh_align",
    "facemesh_register",
    "facemesh_slice_depth",
    "facemesh_slice_height",
    "facemesh_nearest_vertex",
    "facemesh_nearest_point",
    "facemesh_nearest_barycentric",
    "facemesh_from_ssm",
    "ssm_target_points",
    "cc_target_shape",
    "cc_target_shape_legacy"
    ]

current_dir = os.path.dirname(__file__)
DLIB_PREDICTOR_PATH = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")
DLIB_DETECTOR = None
DLIB_PREDICTOR = None


def facemesh_plot (mesh, ax, *, rotation=0, persp="ortho", **kwargs):
    try:
        vertices = mesh.vertices
        triangles = mesh.faces
    except AttributeError:
        vertices, triangles = mesh

    kwargs.setdefault("color", "peachpuff")
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("linewidth", 0)
    kwargs.setdefault("shade", True)
    kwargs.setdefault("antialiased", False)
    kwargs.setdefault("lightsource", LightSource(azdeg=0, altdeg=15))

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


def facemesh_landmarks (mesh):
    global DLIB_DETECTOR, DLIB_PREDICTOR
    if DLIB_DETECTOR is None:
        DLIB_DETECTOR = dlib.get_frontal_face_detector()
    if DLIB_PREDICTOR is None:
        DLIB_PREDICTOR = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

    try:
        vertices = mesh.vertices
        triangles = mesh.faces
    except AttributeError:
        vertices, triangles = mesh

    fig = plt.figure(facecolor='k', dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    facemesh_plot(mesh, ax)
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


def facemesh_align (source_mesh, target_mesh, source_landmarks, target_landmarks):
    matrix0, _, _ = procrustes(source_mesh.vertices[source_landmarks],
                               target_mesh.vertices[target_landmarks])
    matrix, transformed, _ = icp(source_mesh.vertices,
                        target_mesh.vertices,
                        initial = matrix0)
    aligned = Trimesh(vertices=transformed, faces=source_mesh.faces, process=False)
    return aligned, matrix


def facemesh_register (source_mesh, target_mesh, source_landmarks, target_landmarks, k=2.0):
    _, dist, _ = closest_point(target_mesh, source_mesh.vertices)
    dt = np.median(dist) * k
    nrr_points = nricp_amberg(source_mesh, target_mesh,
                            source_landmarks = np.asarray(source_landmarks, dtype=int),
                            target_positions = target_mesh.vertices[target_landmarks],
                            distance_threshold = dt,
                            steps = None)
    return nrr_points


def facemesh_slice_depth (mesh, k=0.2):
    x_verts = mesh.vertices[:,0]
    imin, imax = x_verts.argmin(), x_verts.argmax()
    z_max = mesh.bounds[1,2]
    z_min = (mesh.vertices[imin,imax,2] + mesh.vertices[imax,2])/2.0
    depth = z_max - z_min

    return mesh.slice_plane([0,0,z_min + k*depth], [0,0,1])


def facemesh_slice_height (mesh, landmarks, k=0.2):
    lm_verts = mesh.vertices[landmarks]
    y_min = lm_verts[:,1].min()
    y_max = lm_verts[:,1].max()
    height = y_max - y_min

    sliced = mesh.slice_plane([0,y_min - k*height,0], [0,1,0])\
                  .slice_plane([0,y_max + k*height,0], [0,-1,0])

    _, new_landmarks = sliced.nearest.vertex(lm_verts)

    return sliced, new_landmarks


def facemesh_nearest_vertex (mesh, points):
    _, indices = mesh.nearest.vertex(points)
    return indices


def facemesh_nearest_point (mesh, points):
    closest, _, _ = closest_point(mesh, points)
    return closest


def facemesh_nearest_barycentric (mesh, points):
    closest, _, tridx = closest_point(mesh, points)
    barycentric = points_to_barycentric(mesh.triangles[tridx], closest)
    return (tridx, barycentric)


def facemesh_from_ssm (ssm):
    return Trimesh(ssm.vertices, ssm.triangles_only, process = False)


def ssm_target_points (ssm, targets, indices=None, landmarks=None, minimize="error", iterations=1, wz=0.0, wl=1.0):
    deltas = np.concatenate([ssm.gs_deltas, ssm.ga_deltas], axis=2)
    verts0 = ssm.vertices
    verts = np.copy(verts0)

    nv = verts0.shape[0]

    if indices is None: indices = np.arange(nv)
    verts[indices] = targets

    weights = np.zeros(nv, dtype=np.float32)
    z_min, z_max = np.min(targets[:,2]), np.max(targets[:,2])
    weights[indices] = np.power((targets[:,2] - z_min)/(z_max-z_min), wz)
    if landmarks is not None:
        weights[indices[landmarks]] *= wl

    deltas *= weights[:,np.newaxis,np.newaxis]
    verts *= weights[:,np.newaxis]
    verts0 *= weights[:,np.newaxis]

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
        exyz = np.kron(weights, np.eye(3)).T

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

        verts = (1+k)*verts.dot(R.T) + np.kron(weights,[x,y,z]).reshape(-1,3)
        xt, yt, zt, at, bt, ct, kt = xt+x, yt+y, zt+z, at+a, bt+b, ct+c, kt+k

    fit = Dinv.dot((verts - verts0).flatten())

    weighted_error = np.sum((verts0 + np.dot(deltas,fit) - verts)**2)

    transformation = compose_matrix(scale=(kt,kt,kt), angles=(at,bt,ct), translate=(xt,yt,zt))

    return (ssm.gs_data + fit[:ssm.GS],
            ssm.ga_data + fit[ssm.GS:],
            transformation,
            weighted_error)


def cc_target_shape (character_creator, target_shape,
                     *, preset=None, mode=0, maxiter=100, **kwargs):
    cc = character_creator
    sequence = cc.sequence
    available = [key for tab in cc.tabs.values() for key in tab]

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
    if preset is None: preset = cc.preset
    sam = preset.copy()
    if cc.all_at_once:
        prekeys = (10,11,14,20,21,24,30,31,34)
        preseq = [key for key in sequence if key in prekeys]
    else:
        preseq = []
    for key in preseq:
        value = cc.values[key]
        cc.set_control(key, value, sam)
    s0 = Nt.dot(sam.gs_data)

    # faster sequence, with clip
    def apply_seq (p):
        return np.clip(s0 + mfit.dot(p), *cc.shape_sym_range)

    smin, smax = cc.shape_sym_range
    def in_range (p):
        s = apply_seq(p)
        onezero = np.where((s>=smin) & (s <=smax), 1.0, 0.0)
        return np.diag(onezero)

    if mode <= 0: # minimize shape difference
        def residual (p):
            s = apply_seq(p)
            return s - target_shape
        def jacobian (p):
            return in_range(p).dot(mfit)

    elif mode == 1: # minimize features difference
        def residual (p):
            s = apply_seq(p)
            return np.dot(cc.lgs_coeffs, s - target_shape)
        def jacobian (p):
            return cc.lgs_coeffs.dot(in_range(p)).dot(mfit)

    elif mode >= 2: # minimize vertex difference
        deltas = sam.gs_deltas.reshape(-1, sam.gs_data.size)
        def residual (p):
            s = apply_seq(p)
            return np.dot(deltas, s - target_shape)
        def jacobian (p):
            return deltas.dot(in_range(p)).dot(mfit)

    # sliders bounds
    ranges = np.array([cc.sliders[key].available_range for key in lgs_avail],
                            dtype=np.float32)
    ranges.sort(axis=1)

    # initial guess
    p0 = np.linalg.pinv(mfit).dot(target_shape-s0)
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
    sam.gs_data = apply_seq(result.x)

    for key in preseq:
        solution[key] = cc.values[key]

    result.nit = result.nfev
    return (solution, sam.gs_data, result)


def cc_target_shape_legacy (character_creator, target_shape,
                            *,preset=None, mode=0, maxiter=100,**kwargs):
    cc = character_creator
    sequence = cc.sequence
    available = [key for tab in cc.tabs.values() for key in tab]

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
    if preset is None: preset = cc.preset
    sam = preset.copy()
    if cc.all_at_once:
        prekeys = (10,11,14,20,21,24,30,31,34)
        preseq = [key for key in sequence if key in prekeys]
    else:
        preseq = []
    for key in preseq:
        value = cc.values[key]
        cc.set_control(key, value, sam)
    s0 = Nt.dot(sam.gs_data)

    # faster sequence, without clip
    def apply_seq (p):
        return s0 + mfit.dot(p)

    if mode <= 0: # minimize shape difference
        def residual (p):
            s = apply_seq(p)
            return s - target_shape
        def gradient (p):
            return residual(p).dot(mfit)

    elif mode == 1: # minimize features difference
        def residual (p):
            s = apply_seq(p)
            return np.dot(cc.lgs_coeffs, s - target_shape)
        def gradient (p):
            return residual(p).dot(cc.lgs_coeffs).dot(mfit)

    elif mode >= 2: # minimize vertex difference
        deltas = sam.gs_deltas.reshape(-1, sam.gs_data.size)
        def residual (p):
            s = apply_seq(p)
            return np.dot(deltas, s - target_shape)
        def gradient (p):
            return residual(p).dot(deltas).dot(mfit)

    # avoid shape data saturation
    smin, smax = cc.shape_sym_range
    upper_lim = {"type": "ineq", "fun": lambda p: smax - apply_seq(p).max()}
    lower_lim = {"type": "ineq", "fun": lambda p: apply_seq(p).min() - smin}

    # sliders bounds
    ranges = np.array([cc.sliders[key].available_range for key in lgs_avail],
                            dtype=np.float32)
    ranges.sort(axis=1)

    # initial guess
    p0 = np.linalg.pinv(mfit).dot(target_shape-s0)
    if (np.any(p0 < ranges[:,0]) or
        np.any(p0 > ranges[:,1]) or
        upper_lim["fun"](p0) < 0 or
        lower_lim["fun"](p0) < 0):
        p0 = np.array([cc.values[k] for k in lgs_avail], dtype=np.float32)

    # optimization
    kwargs.setdefault("method","SLSQP")
    kwargs.setdefault("options",{"maxiter":maxiter})
    result = optimize.minimize(lambda p: 0.5*np.sum(residual(p)**2),
                               p0,
                               jac=gradient,
                               bounds=ranges,
                               constraints=[lower_lim, upper_lim],
                               **kwargs)

    solution = dict(zip(lgs_avail, result.x))
    sam.gs_data = apply_seq(result.x)
    sam.gs_data.clip(*cc.shape_sym_range, sam.gs_data)

    for key in preseq:
        solution[key] = cc.values[key]

    result.cost = result.fun
    return (solution, sam.gs_data, result)
