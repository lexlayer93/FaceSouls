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
parser_f2c.add_argument("--mode", default=0, type=int, help="Fit mode")
parser_f2c.add_argument("--maxiter", default=100, type=int, help="Max. iterations")

parser_f2f = subparsers.add_parser("fg2fg", help="Face translation between character creators.")
parser_f2f.add_argument("fg1", help="Source face (.fg)")
parser_f2f.add_argument("cc1", help="Source character creator set (.zip)")
parser_f2f.add_argument("cc2", help="Target character creator set (.zip)")

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
    sliced1 = facemesh_slice_depth(mesh1, k=0.25)
    sliced2 = facemesh_slice_depth(mesh2, k=0.25)
    lm1 = facemesh_landmarks(sliced1)[:60]
    lm2 = facemesh_landmarks(sliced2)[:60]
    lm1 = facemesh_nearest_vertex(mesh1, sliced1.vertices[lm1])
    lm2 = facemesh_nearest_vertex(mesh2, sliced2.vertices[lm2])

    face1.load_data(args.fg1)
    face2.load_data(args.fg1)
    mesh1 = facemesh_from_model(face1)
    mesh2 = facemesh_from_model(face2)
    cropped1 = facemesh_crop(mesh1, lm1, tol=(0.05, 0.1, 0.05))
    cropped2 = facemesh_crop(mesh2, lm2, tol=(0.05, 0.1, 0.05))
    lm1c = facemesh_nearest_vertex(cropped1, mesh1.vertices[lm1])
    lm2c = facemesh_nearest_vertex(cropped2, mesh2.vertices[lm2])

    cropped1, _ = facemesh_align(cropped1, cropped2, lm1c, lm2c)
    targets = facemesh_register(cropped2, cropped1, lm2c, lm1c, k=2.0)

    indices = facemesh_nearest_vertex(mesh2, cropped2.vertices)
    face3, _, _ = model_fit_points(face2, targets, indices, lm2,
                                   wl=2.0, wz=0.2, iterations=2)
    face3.ga_data.fill(0.0)

    fig = plt.figure(figsize=plt.figaspect(1/3), facecolor='k', dpi=100)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("Source", color='w')
    facemesh_plot(mesh1, ax)
    x, y, z = mesh1.vertices[lm1,:].T
    ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("Same data", color='w')
    facemesh_plot(mesh2, ax)
    x, y, z = mesh2.vertices[lm2,:].T
    ax.scatter(x, y, z+1e-3, color='red', marker='o')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("Similar geometry", color='w')
    facemesh_plot((face3.vertices, face3.triangles_only), ax)
    x, y, z = face3.vertices[lm2,:].T
    ax.scatter(x, y, z+1e-3, color='red', marker='o')

    plt.show()


