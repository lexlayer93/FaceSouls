import argparse
from facesouls.character import CharacterCreator
from facesouls.tools import cc_target_shape, facemesh_plot
import matplotlib.pyplot as plt

def match_ext (lof, *ext):
    return len(lof) == len(ext) and all(f.endswith(e) for f,e in zip(lof,ext))

parser = argparse.ArgumentParser()

parser.add_argument("files", nargs='+')
parser.add_argument("--mode", default=0, type=int, help="Fit mode")
parser.add_argument("--maxiter", default=100, type=int, help="Max. iterations")

args = parser.parse_args()

if match_ext(args.files, ".fg", ".zip"):
    cc = CharacterCreator(args.files[1])
    target = cc.models[0].copy()
    target.load_data(args.files[0])

    solution, data, info = cc_target_shape(cc, target.gs_data, mode=args.mode, maxiter=args.maxiter)
    cc.models[0].gs_data = data

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
    facemesh_plot((cc.models[0].vertices, cc.models[0].triangles_only), ax)

    plt.show()
