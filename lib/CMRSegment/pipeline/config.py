import dataclasses
from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.common.constants import ROOT_DIR, LIB_DIR


@dataclasses.dataclass
class PipelineModuleConfig:
    overwrite: bool = False


@dataclasses.dataclass
class SegmentorConfig(PipelineModuleConfig):
    model_path: Path = None
    segment_cine: bool = True
    torch: bool = True

    def __post_init__(self):
        if self.model_path is None:
            raise TypeError("__init__ missing 1 required argument: 'model_path'")


@dataclasses.dataclass
class RefinerConfig(PipelineModuleConfig):
    atlas_dir: Path = None
    n_top_atlas: int = 3

    def __post_init__(self):
        if self.atlas_dir is None:
            raise TypeError("__init__ missing 1 required argument: 'atlas_dir'")


@dataclasses.dataclass
class MeshExtractorConfig(PipelineModuleConfig):
    iso_value: int = 120
    blur: int = 2


@dataclasses.dataclass
class CoregisterConfig(PipelineModuleConfig):
    param_dir: Path = None
    template_dir: Path = None

    def __post_init__(self):
        if self.param_dir is None:
            self.param_dir = LIB_DIR.joinpath("CMRSegment", "resource")
        if self.template_dir is None:
            self.template_dir = ROOT_DIR.joinpath("input", "params")


@dataclasses.dataclass
class MotionTrackerConfig(PipelineModuleConfig):
    param_dir: Path = None
    template_dir: Path = None

    def __post_init__(self):
        if self.param_dir is None:
            self.param_dir = LIB_DIR.joinpath("CMRSegment", "resource")
        if self.template_dir is None:
            self.template_dir = ROOT_DIR.joinpath("input", "params")


class PipelineConfig:
    def __init__(
        self,
        segment: bool,
        refine: bool,
        extract: bool,
        coregister: bool,
        track_motion: bool,
        output_dir: Path,
        overwrite: bool = False,
        model_path: Path = None,
        segment_cine: bool = None,
        torch: bool = True,
        refine_atlas_dir: Path = None,
        refine_n_top: int = 3,
        iso_value: int = 120,
        blur: int = 2,
        param_dir: Path = None,
        template_dir: Path = None,
        use_irtk: bool = False
    ):
        self.output_dir = output_dir
        self.overwrite = overwrite
        if segment:
            self.segment_config = SegmentorConfig(
                model_path=model_path,
                segment_cine=segment_cine,
                overwrite=overwrite,
                torch=torch,

            )
            self.segment = True
        else:
            self.segment = False
            self.segment_config = None
        if refine:
            assert refine_atlas_dir is not None, "Need to provide atlas directory if using refine."
            self.refine = True
            self.refine_config = RefinerConfig(
                atlas_dir=refine_atlas_dir,
                n_top_atlas=refine_n_top,
            )
        else:
            self.refine = False
            self.refine_config = None
        if extract:
            self.extract_config = MeshExtractorConfig(
                iso_value=iso_value, blur=blur,  overwrite=overwrite
            )
            self.extract = True
        else:
            self.extract = False
            self.extract_config = None
        if coregister:
            self.coregister_config = CoregisterConfig(
                overwrite=overwrite, param_dir=param_dir, template_dir=template_dir
            )
            self.coregister = True
        else:
            self.coregister = False
            self.coregister_config = None
        self.use_irtk = use_irtk

        if track_motion:
            self.motion_tracker_config = MotionTrackerConfig(
                overwrite=overwrite, param_dir=param_dir, template_dir=template_dir
            )
            self.track_motion = True
        else:
            self.track_motion = False
            self.motion_tracker_config = None

    @staticmethod
    def argument_parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-o", "--output-dir", dest="output_dir", required=True, type=str)
        parser.add_argument("--overwrite", dest="overwrite", action="store_true")
        parser.add_argument("--irtk", dest="use_irtk", action="store_true")

        parser.add_argument(
            "--segment", dest="segment", action="store_true",
            help="Run segmentor module to segment image."
        )
        parser.add_argument(
            "--refine", dest="refine", action="store_true",
            help="Run refiner module to refine segmentation using multi-altas registration."
        )
        parser.add_argument(
            "--extract", dest="extract", action="store_true",
            help="Extract meshes from segmentations."
        )
        parser.add_argument("--coregister", dest="coregister", action="store_true")
        parser.add_argument("--track-motion", dest="track_motion", action="store_true")

        segment_parser = parser.add_argument_group("segment")
        segment_parser.add_argument("--model-path", dest="model_path", default=None, type=str)
        segment_parser.add_argument("--segment-cine", action="store_true")
        segment_parser.add_argument("--torch", dest="torch", action="store_true")

        segment_parser = parser.add_argument_group("refine")
        segment_parser.add_argument(
            "--atlas-dir", dest="refine_atlas_dir", type=str, default=None,
            help="Directory where all the atlases will be used for refinement."
        )
        segment_parser.add_argument(
            "--n-top", dest="refine_n_top", type=int, default=3,
            help="Number of top similar atlases, selected for refinement"
        )

        extract_parser = parser.add_argument_group("extract")
        extract_parser.add_argument("--iso-value", dest="iso_value", default=120, type=int)
        extract_parser.add_argument("--blur", dest="blur", default=2, type=int)

        coregister_parser = parser.add_argument_group("coregister")
        coregister_parser.add_argument("--template-dir", dest="template_dir", default=None, type=str)
        coregister_parser.add_argument("--param-dir", dest="param_dir", default=None, type=str)

        return parser

    @classmethod
    def from_args(cls, args):
        return cls(
            segment=args.segment,
            refine=args.refine,
            extract=args.extract,
            coregister=args.coregister,
            track_motion=args.track_motion,
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
            model_path=Path(args.model_path) if args.model_path is not None else None,
            segment_cine=args.segment_cine,
            torch=args.torch,
            refine_atlas_dir=Path(args.refine_atlas_dir) if args.refine_atlas_dir is not None else None,
            refine_n_top=args.refine_n_top,
            iso_value=args.iso_value,
            blur=args.blur,
            param_dir=Path(args.param_dir) if args.param_dir is not None else None,
            template_dir=Path(args.template_dir) if args.template_dir is not None else None,
            use_irtk=args.use_irtk,
        )
