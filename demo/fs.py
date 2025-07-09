import argparse
from facesouls.character import CharacterCreator
from facesouls.tools import *
import matplotlib.pyplot as plt


def fg2cc (cc, src, dst=None,
           *, preset=None, mode=2, maxiter=100,
           hide=False, how=False, step=5):
    cc = CharacterCreator(cc)
    target = cc.models[0].copy()
    target.load_data(src)
    if isinstance(preset, str):
        if preset.endswith(".fg"):
            cc.load_data(preset)
        elif preset.endswith(".csv"):
            cc.load_values(preset)

    solution, replica, info = cc_fit_shape(cc, target, mode=mode, maxiter=maxiter)
    replica.ga_data.fill(0.0)

    out = f"# {info.message}"
    out += f"\n# Iterations: {info.nit}"
    out += f"\n# Error({mode}): {info.cost}"

    if how:
        lj = len(max(cc.labels.values(), key=len))
        def howtoset (value, vmin, vmax, *, step=1):
            miss1 = (value-vmin) % step
            miss1 = min(miss1, step-miss1)
            miss2 = (vmax-value) % step
            miss2 = min(miss2, step-miss2)
            if miss1 <= miss2:
                k = round((value-vmin)/step)
                return '+' + str(k)
            else:
                k = round((vmax-value)/step)
                return '-' + str(k)

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

    if hide:
        return

    fig = plt.figure(figsize=plt.figaspect(0.5), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot((target.vertices, target.triangles_only), ax)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Replica", color='w')
    facemesh_plot((replica.vertices, replica.triangles_only), ax)

    plt.show()


def fg2fg (cc1, cc2, src, dst=None,
           *, light=15, depth=0.0, height=0.1, dist=2.0, unsafe=False,
           weight_lm=2.0, weight_z=0.2, iterations=2, minimize="error",
           hide=False, show_lm=False, rotate=0):
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

    face3, _, err = model_fit_points(face2, targets, indices, lm2,
                                     wl=weight_lm, wz=weight_z,
                                     iterations=iterations,
                                     minimize=minimize)
    face3.ga_data.fill(0.0)

    if dst is not None:
        face3.save_data(dst)

    print("Error:", err)

    if hide:
        return

    fig = plt.figure(figsize=plt.figaspect(1), facecolor='k', dpi=200)

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot(mesh1, ax, rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = mesh1.vertices[lm1,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("Same data", color='w')
    facemesh_plot(mesh2, ax, rotation=rotate, light_alt=light)
    if show_lm:
        x, y, z = mesh2.vertices[lm2,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

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

    parser_f2c = subparsers.add_parser("fg2cc", help="Face replication for a character creator.")
    parser_f2c.add_argument("cc", help="Character creator set (.zip).")
    parser_f2c.add_argument("src", help="Input, target face (.fg).")
    parser_f2c.add_argument("dst", nargs='?', default=None, help="Output, optimal sliders (.csv).")
    parser_f2c.add_argument("--preset", default=None, help="Preset face (.fg/.csv).")
    parser_f2c.add_argument("--mode", default=2, type=int, help="Fit mode.")
    parser_f2c.add_argument("--maxiter", default=100, type=int, help="Max. iterations.")
    parser_f2c.add_argument("--hide", action="store_true", help="Do not show target/replica faces.")
    parser_f2c.add_argument("--how", action="store_true", help="Output slider as ticks from sides.")
    parser_f2c.add_argument("--step", default=5, type=int, help="Steps by tick (must use with --how).")

    parser_f2f = subparsers.add_parser("fg2fg", help="Face translation between character creators.")
    parser_f2f.add_argument("cc1", help="Source character creator set (.zip).")
    parser_f2f.add_argument("cc2", help="Destined character creator set (.zip).")
    parser_f2f.add_argument("src", help="Input, target face (.fg).")
    parser_f2f.add_argument("dst", nargs='?', default=None, help="Output, optimal face (.fg).")
    parser_f2f.add_argument("--light", type=float, default=15.0, help="Light source elevation.")
    parser_f2f.add_argument("--depth", type=float, default=0.0, help="Face outline, further ahead (+) or further back (-).")
    parser_f2f.add_argument("--height", type=float, default=0.1, help="Tolerance of vertical cropping.")
    parser_f2f.add_argument("--dist", type=float, default=2.0, help="Distance threshold for face register.")
    parser_f2f.add_argument("--weight_lm", type=float, default=2.0, help="Landmarks relative weight.")
    parser_f2f.add_argument("--weight_z", type=float, default=0.2, help="Z-weight control.")
    parser_f2f.add_argument("--iterations", type=int, default=2, help="Number of fit iterations.")
    parser_f2f.add_argument("--minimize", choices=["error", "effort"], default="error", help="Z-weight control.")
    parser_f2f.add_argument("--unsafe", action="store_true", help="Unsafe landmark detection.")
    parser_f2f.add_argument("--hide", action="store_true", help="Do not show target/replica faces.")
    parser_f2f.add_argument("--show_lm", action="store_true", help="Show landmarks (do not use with --hide).")
    parser_f2f.add_argument("--rotate", type=float, default=0.0, help="90 for side view.")

    args = parser.parse_args()

    args = vars(args)
    cmd = args.pop("command")
    if cmd == "howto":
        howto(**args)
    elif cmd == "fg2cc":
        fg2cc(**args)
    elif cmd == "fg2fg":
        fg2fg(**args)
