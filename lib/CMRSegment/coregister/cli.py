from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.coregister import Coregister
from CMRSegment.common.resource import Mesh, Segmentation, Phase


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mesh-dir", dest="mesh_dir", type=str, required=True)
    parser.add_argument("--phase", dest="phase", type=str, required=True)
    parser.add_argument("--segmentation-path", dest="segmentation_path", type=str, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--landmark-path", dest="landmark_path", type=str, required=True)
    parser.add_argument("--template-params-dir", dest="params_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = Mesh(
        phase=Phase[args.phase],
        dir=Path(args.mesh_dir)
    )
    params_dir = Path(args.params_dir)
    coregister = Coregister(
        template_dir=params_dir,
        param_dir=params_dir
    )
    landmark_path = Path(args.landmark_path)
    segmentation = Segmentation(
        phase=Phase[args.phase],
        path=Path(args.segmentation_path)
    )
    coregister.run(
        mesh=mesh, segmentation=segmentation, landmark_path=landmark_path, output_dir=Path(args.output_dir)
    )


if __name__ == '__main__':
    main()
