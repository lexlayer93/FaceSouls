import argparse
from facesouls.character import CharacterCreator
from facesouls.facemesh import Facemesh
from facesouls.tools import *
import matplotlib.pyplot as plt


def view (cc, face):
    cc = CharacterCreator.fromzip(cc)
    if isinstance(face, str):
        if face.endswith(".fg"):
            cc.load_data(face)
        else:
            cc.load_values(face)
    face = cc.models[0]
    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    faceplot(ax, face.vertices, face.triangles)
    plt.show()


def diff (cc, key, length=1.0):
    cc = CharacterCreator.fromzip(cc)
    face = cc.models[0]
    cc.set_zero(face)
    tails = face.vertices.T

    val0 = cc.get_control(key)
    cc.set_control(key, val0+1.0, face)
    arrows = face.vertices.T - tails

    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)
    ax = fig.add_subplot(projection="3d")
    ax.set_title(cc.sliders[key].debug_label, color='w')

    faceplot(ax, face.vertices0, face.triangles_only)
    ax.quiver(*tails, *arrows, length=length)

    plt.show()


def fg2cc (
    cc, src, dst=None, *, preset=None,
    mode=2, maxit=100,
    how=False, step=5,
    show=False
    ):
    cc = CharacterCreator.fromzip(cc)
    target = cc.face.copy()
    target.load_data(src)
    if isinstance(preset, str):
        if preset.endswith(".fg"):
            cc.load_data(preset)
        else:
            cc.load_values(preset)

    solution, replica, info = facegen_to_cc(
        target, cc,
        mode=mode, maxiter=maxit)

    if how:
        lj = len(max(cc.labels.values(), key=len))
        def _howtoset (value, vmin, vmax, *, step=1):
            miss1 = (value-vmin) % step
            miss1 = min(miss1, step-miss1)
            miss2 = (vmax-value) % step
            miss2 = min(miss2, step-miss2)
            tck1 = round((value-vmin)/step)
            tck2 = round((vmax-value)/step)
            if miss1 < miss2 or (miss1 == miss2 and tck1 <= tck2):
                return '+' + str(tck1)
            else:
                k = round((vmax-value)/step)
                return '-' + str(tck2)

    out = f"# {info.message}"
    out += f"\n# Iterations: {info.nit}"
    out += f"\n# Error({mode}): {info.cost}"

    for key, value in solution.items():
        slider = cc.sliders[key]
        tab = slider.tab
        lab = slider.label
        val = slider.float2int(value)
        if not how:
            out += f"\n{key:03d}, {tab}, {lab}, {val};"
        else:
            tck = _howtoset(val, *slider.int_range, step=step)
            out += f"\n{tck:<4} {lab.ljust(lj)} {tab}"

    if dst is None:
        print(out)
    else:
        with open(dst, 'w') as f: f.write(out)

    if not show:
        return

    fig = plt.figure(figsize=plt.figaspect(1/2), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    faceplot(ax, target.vertices, target.triangles)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Replica", color='w')
    faceplot(ax, replica.vertices, replica.triangles)

    plt.show()


def _set_landmarks (mesh, *, depth, light):
    sliced = mesh.sliced(depth)
    sliced.landmarks = find_landmarks(sliced, light_alt=light)[:60]
    mesh.landmarks = mesh.nearest_vertex(sliced.keypoints)

def _register (
    source, target, *,
    height, dist, ws, wn, force):
    def _steps (ws=0.01, wl=10, wn=0.5, max_iter=10, n=4):
        steps = [None]*n
        for i in range(n-1):
            steps[i] = (ws*(i+1), wl/2**i, wn, max_iter)
        steps[n-1] = (ws, 0.0, 0.0, max_iter)
        return steps

    cropped_src = source.cropped(ytol=height)
    cropped_tgt = target.cropped(ytol=height)

    aligned_tgt = cropped_tgt.aligned(cropped_src)

    dt = cropped_src.distance(aligned_tgt)*(2**dist)
    fitted = cropped_src.fitted(
        aligned_tgt,
        distance_threshold=dt,
        steps=_steps(ws=ws, wn=wn),
        force=force)
    indices = source.nearest_vertex(cropped_src.vertices)

    return fitted, indices


def _plot (data, nr, nc):
    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)
    for idx, title, verts, tris, lms, light_alt, rotation, show_lm in data:
        ax = fig.add_subplot(nr, nc, idx, projection="3d")
        ax.set_title(title, color="w")
        faceplot(ax, verts, tris, rotation=rotation, light_alt=light_alt)
        if show_lm:
            x, y, z = verts[lms,:].T
            ax.scatter(x, y, z+1e-3, color="red", marker='x')
    plt.show()


def fg2fg (
    cc1, cc2, src, dst=None, *,
    light=25, depth=0.0,
    height=0.1, dist=1.0,
    ws=0.01, wn=0.5,
    wl=1.0, wz=0.0,
    nit=2, minimize="error",
    force=False, sym=False,
    show=False, show_lm=False, rotate=0
    ):
    cc1 = CharacterCreator.fromzip(cc1)
    face1 = cc1.models[0].copy()
    cc1.set_zero(face1)
    mesh1 = Facemesh.fromfg(face1)
    _set_landmarks(mesh1, depth=depth, light=light)
    face1.load_data(src)
    mesh1.vertices = face1.vertices

    cc2 = CharacterCreator.fromzip(cc2)
    face2 = cc2.models[0].copy()
    cc2.set_zero(face2)
    mesh2 = Facemesh.fromfg(face2)
    _set_landmarks(mesh2, depth=depth, light=light)
    face2.load_data(src)
    mesh2.vertices = face2.vertices

    fitted, indices = _register(
        mesh2, mesh1,
        height=height, dist=dist,
        ws=ws, wn=wn,
        force=force)

    facefit, err = facemesh_to_fg(
        fitted, face2,
        indices=indices,
        landmarks=mesh2.landmarks,
        wl=wl, wz=wz,
        iterations=nit,
        minimize=minimize,
        asymmetry=not sym)

    if dst is not None:
        facefit.save_data(dst)

    print("# Error:", err)
    print("# Solution:", facefit.gs_data)

    if not show:
        return

    plots = [
        (1, "Target", mesh1.vertices, mesh1.triangles, mesh1.landmarks, light, rotate, show_lm),
        (2, "Closest", facefit.vertices, facefit.triangles, mesh2.landmarks, light, rotate, show_lm),
        (3, "Same data", mesh2.vertices, mesh2.triangles, mesh2.landmarks, light, rotate, show_lm),
        (4, "Deformation", fitted.vertices, fitted.triangles, fitted.landmarks, light, rotate, show_lm)
        ]

    _plot(plots, 2, 2)


def obj2fg (
    cc, src, dst=None, *,
    light=25, depth=0.0,
    height=0.1, dist=1.0,
    ws=0.01, wn=0.5,
    wl=1.0, wz=0.0,
    nit=1, minimize="error",
    force=False, sym=False,
    show=False, show_lm=False, rotate=0
    ):
    cc = CharacterCreator.fromzip(cc)
    cc.set_zero(cc.face)
    mesh1 = Facemesh.fromfile(src)
    mesh2 = Facemesh.fromfg(cc.face)

    _set_landmarks(mesh1, depth=depth, light=light)
    _set_landmarks(mesh2, depth=depth, light=light)

    fitted, indices = _register(
        mesh2, mesh1,
        height=height, dist=dist,
        ws=ws, wn=wn,
        force=force)

    facefit, err = facemesh_to_fg(
        fitted, cc.face,
        indices=indices,
        landmarks=mesh2.landmarks,
        wl=wl, wz=wz,
        iterations=nit,
        minimize=minimize,
        asymmetry=not sym)

    if dst is not None:
        facefit.save_data(dst)

    print("# Error:", err)
    print("# Solution:", facefit.gs_data)

    if not show:
        return

    plots = [
        (1, "Target", mesh1.vertices, mesh1.triangles, mesh1.landmarks, light, rotate, show_lm),
        (2, "Closest", facefit.vertices, facefit.triangles, mesh2.landmarks, light, rotate, show_lm),
        (3, "Model", mesh2.vertices, mesh2.triangles, mesh2.landmarks, light, rotate, show_lm),
        (4, "Deformation", fitted.vertices, fitted.triangles, fitted.landmarks, light, rotate, show_lm)
        ]

    _plot(plots, 2, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_view = subparsers.add_parser("view", help="View face 3D model.")
    parser_view.add_argument("cc", help="Character creator set (.zip).")
    parser_view.add_argument("face", help="Face (.fg/.csv).")

    parser_diff = subparsers.add_parser("diff", help="View control diff morph.")
    parser_diff.add_argument("cc", help="Character creator set (.zip).")
    parser_diff.add_argument("key", type=int, help="Control key id.")
    parser_diff.add_argument("--length", type=float, default=1.0, help="Arrows length.")

    parser_f2c = subparsers.add_parser("fg2cc", help="Find slider values to match a face inside character creator.")
    parser_f2c.add_argument("cc", help="Character creator set (.zip).")
    parser_f2c.add_argument("src", help="Input, target face (.fg).")
    parser_f2c.add_argument("dst", nargs='?', default=None, help="Output, optimal sliders (.csv).")
    parser_f2c.add_argument("--preset", default=None, help="Preset face (.fg/.csv), required for DeS/DS1.")
    parser_f2c.add_argument("--mode", default=2, type=int, help="Fit mode (0 = data, 1 = features, 2 = geometry).")
    parser_f2c.add_argument("--maxit", default=100, type=int, help="Max. number of iterations.")
    parser_f2c.add_argument("--how", action="store_true", help="Output slider as ticks from sides.")
    parser_f2c.add_argument("--step", default=5, type=int, help="Steps by tick (must use with --how).")
    parser_f2c.add_argument("--show", action="store_true", help="Show target/replica faces.")

    parser_f2f = subparsers.add_parser("fg2fg", help="Convert a face to another character creator.")
    parser_f2f.add_argument("cc1", help="Source character creator set (.zip).")
    parser_f2f.add_argument("cc2", help="Destined character creator set (.zip).")
    parser_f2f.add_argument("src", help="Input, target face (.fg).")
    parser_f2f.add_argument("dst", nargs='?', default=None, help="Output, optimal face (.fg).")
    parser_f2f.add_argument("--light", type=float, default=25.0, help="Light source elevation.")
    parser_f2f.add_argument("--depth", type=float, default=0.0, help="Face outline, further ahead (+) or further back (-).")
    parser_f2f.add_argument("--height", type=float, default=0.2, help="Vertical cropping tolerance, for mesh registration.")
    parser_f2f.add_argument("--dist", type=float, default=1.0, help="Distance threshold exponent, for mesh registration.")
    parser_f2f.add_argument("--ws", type=float, default=0.01, help="Smoothness weight, for mesh registration.")
    parser_f2f.add_argument("--wn", type=float, default=0.1, help="Normals weight control, for mesh registration.")
    parser_f2f.add_argument("--wl", type=float, default=10.0, help="Landmarks relative weight, for mesh conversion.")
    parser_f2f.add_argument("--wz", type=float, default=0.2, help="Depth weight control, for mesh conversion.")
    parser_f2f.add_argument("--nit", type=int, default=2, help="Number of fit iterations, for mesh converson.")
    parser_f2f.add_argument("--minimize", choices=["error", "effort"], default="error", help="Minimize error for accuracy, effort for smoothness.")
    parser_f2f.add_argument("--force", action="store_true", help="Force geometry after registration.")
    parser_f2f.add_argument("--sym", action="store_true", help="Symmetryc only.")
    parser_f2f.add_argument("--show", action="store_true", help="Show target/replica faces.")
    parser_f2f.add_argument("--show_lm", action="store_true", help="Show landmarks (do not use with --hide).")
    parser_f2f.add_argument("--rotate", type=float, default=0.0, help="90 degrees for side view.")

    parser_o2f = subparsers.add_parser("obj2fg", help="Convert a face mesh to FaceGen.")
    parser_o2f.add_argument("cc", help="Source character creator set (.zip).")
    parser_o2f.add_argument("src", help="Input, target mesh (.obj).")
    parser_o2f.add_argument("dst", nargs='?', default=None, help="Output, optimal face (.fg).")
    parser_o2f.add_argument("--light", type=float, default=25.0, help="Light source elevation.")
    parser_o2f.add_argument("--depth", type=float, default=0.0, help="Face outline, further ahead (+) or further back (-).")
    parser_o2f.add_argument("--height", type=float, default=0.2, help="Vertical cropping tolerance.")
    parser_o2f.add_argument("--dist", type=float, default=1.0, help="Distance threshold exponent for mesh registration.")
    parser_o2f.add_argument("--ws", type=float, default=0.01, help="Smoothness weight, for mesh registration.")
    parser_o2f.add_argument("--wn", type=float, default=0.1, help="Normals weight control, for mesh registration.")
    parser_o2f.add_argument("--wl", type=float, default=10.0, help="Landmarks relative weight, for mesh conversion.")
    parser_o2f.add_argument("--wz", type=float, default=0.2, help="Depth weight control, for mesh conversion.")
    parser_o2f.add_argument("--nit", type=int, default=2, help="Number of fit iterations.")
    parser_o2f.add_argument("--minimize", choices=["error", "effort"], default="error", help="Minimize error for accuracy, effort for smoothness.")
    parser_o2f.add_argument("--force", action="store_true", help="Force geometry after registration.")
    parser_o2f.add_argument("--sym", action="store_true", help="Symmetryc only.")
    parser_o2f.add_argument("--show", action="store_true", help="Show target/replica faces.")
    parser_o2f.add_argument("--show_lm", action="store_true", help="Show landmarks (do not use with --hide).")
    parser_o2f.add_argument("--rotate", type=float, default=0.0, help="90 degrees for side view.")

    args = parser.parse_args()

    args = vars(args)
    cmd = args.pop("command")
    if cmd == "view":
        view(**args)
    elif cmd == "diff":
        diff(**args)
    elif cmd == "fg2cc":
        fg2cc(**args)
    elif cmd == "fg2fg":
        fg2fg(**args)
    elif cmd == "obj2fg":
        obj2fg(**args)
    else:
        print(f"Command {cmd} not found.")
