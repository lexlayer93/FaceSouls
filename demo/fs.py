import argparse
from facesouls.character import CharacterCreator
from facesouls.tools import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)

parser_f2c = subparsers.add_parser("fg2cc", help="Face replication for a character creator.")
parser_f2c.add_argument("target", help="Target face (.fg)")
parser_f2c.add_argument("cc", help="Character creator set (.zip)")
parser_f2c.add_argument("preset", default=None, nargs='?', help="Preset face (.fg/.csv)")
parser_f2c.add_argument("--mode", default=2, type=int, help="Fit mode")
parser_f2c.add_argument("--maxiter", default=100, type=int, help="Max. iterations")

parser_f2f = subparsers.add_parser("fg2fg", help="Face translation between character creators.")
parser_f2f.add_argument("fg1", help="Target/original face (.fg)")
parser_f2f.add_argument("cc1", help="Original character creator set (.zip)")
parser_f2f.add_argument("cc2", help="Target character creator set (.zip)")
parser_f2f.add_argument("fg2", help="Replica/output face (.fg)")
parser_f2f.add_argument("--depth", type=float, default=0.0, help="Face outline, further ahead (+) or further back (-).")
parser_f2f.add_argument("--height", type=float, default=0.1, help="Tolerance to crop the face vertically.")
parser_f2f.add_argument("--dt", type=float, default=2.0, help="Distance threshold for face register.")
parser_f2f.add_argument("--wl", type=float, default=2.0, help="Landmarks relative weight.")
parser_f2f.add_argument("--wz", type=float, default=0.2, help="Z-weight control.")
parser_f2f.add_argument("--nit", type=int, default=2, help="Number of fit iterations.")
parser_f2f.add_argument("--min", choices=["error", "effort"], default="error", help="Z-weight control.")
parser_f2f.add_argument("--show_lm", action='store_true', help="Show landmarks")

args = parser.parse_args()

if args.command == "fg2cc":
    cc = CharacterCreator(args.cc)
    target = cc.models[0].copy()
    target.load_data(args.target)
    preset = args.preset
    if isinstance(preset, str):
        if preset.endswith(".fg"):
            cc.load_data(preset)
        elif preset.endswith(".csv"):
            cc.load_values(preset)

    solution, replica, info = cc_fit_shape(cc, target, mode=args.mode, maxiter=args.maxiter)
    replica.ga_data.fill(0.0)

    print("#", info.message)
    print(f"# Iterations:", info.nit)
    print(f"# Error({args.mode}):", info.cost)

    for key, value in solution.items():
        slider = cc.sliders[key]
        tab = slider.tab
        lab = slider.label
        val = slider.float2int(value)
        print(f"{key:03d}, {tab}, {lab}, {val};")

    fig = plt.figure(figsize=plt.figaspect(0.5), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot((target.vertices, target.triangles_only), ax)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Replica", color='w')
    facemesh_plot((replica.vertices, replica.triangles_only), ax)

    plt.show()

elif args.command == "fg2fg":
    cc1 = CharacterCreator(args.cc1)
    cc2 = CharacterCreator(args.cc2)
    face1 = cc1.models[0].copy()
    face2 = cc2.models[0].copy()

    mesh1 = facemesh_from_model(face1)
    mesh2 = facemesh_from_model(face2)
    sliced1 = facemesh_slice_depth(mesh1, k=args.depth)
    sliced2 = facemesh_slice_depth(mesh2, k=args.depth)
    lm1 = facemesh_landmarks(sliced1)[:60]
    lm2 = facemesh_landmarks(sliced2)[:60]
    lm1 = facemesh_nearest_vertex(mesh1, sliced1.vertices[lm1])
    lm2 = facemesh_nearest_vertex(mesh2, sliced2.vertices[lm2])

    face1.load_data(args.fg1)
    face2.load_data(args.fg1)
    mesh1 = facemesh_from_model(face1)
    mesh2 = facemesh_from_model(face2)
    cropped1 = facemesh_crop(mesh1, lm1, y_tol=args.height)
    cropped2 = facemesh_crop(mesh2, lm2, y_tol=args.height)
    lm1c = facemesh_nearest_vertex(cropped1, mesh1.vertices[lm1])
    lm2c = facemesh_nearest_vertex(cropped2, mesh2.vertices[lm2])

    cropped1, _ = facemesh_align(cropped1, cropped2, lm1c, lm2c)
    targets = facemesh_register(cropped2, cropped1, lm2c, lm1c, k=args.dt)

    indices = facemesh_nearest_vertex(mesh2, cropped2.vertices)
    face3, _, _ = model_fit_points(face2, targets, indices, lm2,
                                   wl=args.wl, wz=args.wz,
                                   iterations=args.nit,
                                   minimize=args.min)
    face3.ga_data.fill(0.0)
    face3.save_data(args.fg2)

    fig = plt.figure(figsize=plt.figaspect(1/3), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("Target", color='w')
    facemesh_plot(mesh1, ax)
    if args.show_lm:
        x, y, z = mesh1.vertices[lm1,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("Same data", color='w')
    facemesh_plot(mesh2, ax)
    if args.show_lm:
        x, y, z = mesh2.vertices[lm2,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("Closest geometry", color='w')
    facemesh_plot((face3.vertices, face3.triangles_only), ax)
    if args.show_lm:
        x, y, z = face3.vertices[lm2,:].T
        ax.scatter(x, y, z+1e-3, color='red', marker='o')

    plt.show()


