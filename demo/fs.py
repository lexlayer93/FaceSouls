import argparse
from facesouls.character import CharacterCreator
from facesouls.tools import *
import matplotlib.pyplot as plt


def view (cc, face):
    cc = CharacterCreator(cc)
    if isinstance(face, str):
        if face.endswith(".fg"):
            cc.load_data(face)
        else:
            cc.load_values(face)
    face = cc.models[0]
    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    facemesh_plot((face.vertices, face.triangles_only), ax)
    plt.show()


def diff (cc, key, length=1.0):
    cc = CharacterCreator(cc)
    face = cc.models[0]
    cc.set_zero(face)
    tails = face.vertices.T

    val0 = cc.get_control(key)
    cc.set_control(key, val0+1.0, face)
    arrows = face.vertices.T - tails

    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)
    ax = fig.add_subplot(projection="3d")
    ax.set_title(cc.sliders[key].debug_label, color='w')
    facemesh_plot((face.vertices0, face.triangles_only), ax)
    ax.quiver(*tails, *arrows, length=length)

    plt.show()


def fg2cc (cc, src, dst=None, *, preset=None,
           mode=2, maxit=100,
           how=False, step=5,
           show=False):
    cc = CharacterCreator(cc)
    target = cc.models[0].copy()
    target.load_data(src)
    if isinstance(preset, str):
        if preset.endswith(".fg"):
            cc.load_data(preset)
        else:
            cc.load_values(preset)

    solution, replica, info = fit_cc_shape(cc, target, mode=mode, maxiter=maxit)
    replica.ga_data.fill(0.0)

    if how:
        lj = len(max(cc.labels.values(), key=len))
        def howtoset (value, vmin, vmax, *, step=1):
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
            tck = howtoset(val, *slider.int_range, step=step)
            out += f"\n{tck:<4} {lab.ljust(lj)} {tab}"

    if dst is None:
        print(out)
    else:
        with open(dst, 'w') as f: f.write(out)

    if not show:
        return

    fig = plt.figure(figsize=plt.figaspect(0.5), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot((target.vertices, target.triangles_only), ax)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Replica", color='w')
    facemesh_plot((replica.vertices, replica.triangles_only), ax)

    plt.show()


def fg2fg (cc1, cc2, src, dst=None, *,
           light=25, depth=0.0,
           height=0.1, dist=1.0,
           wl=2.0, wz=0.2, nit=2, minimize="error",
           unsafe=False, force=False,
           show=False, show_lm=False, rotate=0):
    cc1 = CharacterCreator(cc1)
    cc2 = CharacterCreator(cc2)
    face1 = cc1.models[0].copy()
    face2 = cc2.models[0].copy()
    cc1.set_zero(face1)
    cc2.set_zero(face2)

    if unsafe:
        face1.load_data(src)
        face2.load_data(src)

    mesh1 = facemesh_from_model(face1)
    mesh2 = facemesh_from_model(face2)
    sliced1 = facemesh_slice_depth(mesh1, k=depth)
    sliced2 = facemesh_slice_depth(mesh2, k=depth)
    lm1 = facemesh_landmarks(sliced1, light_alt=light)[:60]
    lm2 = facemesh_landmarks(sliced2, light_alt=light)[:60]
    lm1 = facemesh_nearest_vertex(mesh1, sliced1.vertices[lm1])
    lm2 = facemesh_nearest_vertex(mesh2, sliced2.vertices[lm2])

    if not unsafe:
        face1.load_data(src)
        face2.load_data(src)
        mesh1 = facemesh_from_model(face1)
        mesh2 = facemesh_from_model(face2)

    cropped1 = facemesh_crop(mesh1, lm1, y_tol=height)
    cropped2 = facemesh_crop(mesh2, lm2, y_tol=height)
    lm1c = facemesh_nearest_vertex(cropped1, mesh1.vertices[lm1])
    lm2c = facemesh_nearest_vertex(cropped2, mesh2.vertices[lm2])
    cropped1, _ = facemesh_align(cropped1, cropped2, lm1c, lm2c)

    targets = facemesh_register(cropped2, cropped1, lm2c, lm1c, k=dist)
    indices = facemesh_nearest_vertex(mesh2, cropped2.vertices)
    if force:
        targets = facemesh_nearest_point(cropped1, targets)

    face3, _, err = fit_model_points(face2, targets, indices, lm2,
                                     wl=wl, wz=wz,
                                     iterations=nit,
                                     minimize=minimize)
    face3.ga_data.fill(0.0)

    print("Error:", err)

    if dst is not None:
        face3.save_data(dst)

    if not show:
        return

    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot(cropped1, ax, rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = cropped1.vertices[lm1c,:].T
        print(x.min(),x.max())
        ax.scatter(x, y, z+1e-3, color="red", marker='o')

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("Same data", color='w')
    facemesh_plot(cropped2, ax, rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = cropped2.vertices[lm2c,:].T
        print(x.min(),x.max())
        ax.scatter(x, y, z+1e-3, color="red", marker='o')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title("Closest geometry", color='w')
    facemesh_plot((face3.vertices, face3.triangles_only), ax,
                  rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = face3.vertices[lm2,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_title("Deformation", color='w')
    facemesh_plot((targets, cropped2.faces), ax,
                  rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = targets[lm2c,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_view = subparsers.add_parser("view", help="View face 3D model.")
    parser_view.add_argument("cc", help="Character creator set (.zip).")
    parser_view.add_argument("face", help="Face (.fg/.csv).")

    parser_view = subparsers.add_parser("diff", help="View control diff morph.")
    parser_view.add_argument("cc", help="Character creator set (.zip).")
    parser_view.add_argument("key", type=int, help="Control key id.")
    parser_view.add_argument("--length", type=float, default=1.0, help="Arrows length.")

    parser_f2c = subparsers.add_parser("fg2cc", help="Find slider values to match a face inside character creator.")
    parser_f2c.add_argument("cc", help="Character creator set (.zip).")
    parser_f2c.add_argument("src", help="Input, target face (.fg).")
    parser_f2c.add_argument("dst", nargs='?', default=None, help="Output, optimal sliders (.csv).")
    parser_f2c.add_argument("--preset", default=None, help="Preset face (.fg/.csv), only for DeS/DS1.")
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
    parser_f2f.add_argument("--height", type=float, default=0.1, help="Vertical cropping tolerance.")
    parser_f2f.add_argument("--dist", type=float, default=1.0, help="Distance threshold exponent for mesh registration.")
    parser_f2f.add_argument("--wl", type=float, default=2.0, help="Landmarks relative weight.")
    parser_f2f.add_argument("--wz", type=float, default=0.2, help="Z-weight control.")
    parser_f2f.add_argument("--nit", type=int, default=2, help="Number of fit iterations.")
    parser_f2f.add_argument("--minimize", choices=["error", "effort"], default="error", help="Z-weight control.")
    parser_f2f.add_argument("--unsafe", action="store_true", help="Unsafe landmark detection.")
    parser_f2f.add_argument("--force", action="store_true", help="Force geometry after registration.")
    parser_f2f.add_argument("--show", action="store_true", help="Show target/replica faces.")
    parser_f2f.add_argument("--show_lm", action="store_true", help="Show landmarks (do not use with --hide).")
    parser_f2f.add_argument("--rotate", type=float, default=0.0, help="90 degrees for side view.")

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
    else:
        print(f"Command {cmd} not found.")
