from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR.joinpath("models")
DATA_DIR = ROOT_DIR.joinpath("data")
LIB_DIR = Path(__file__).parent.parent.parent
RESOURCE_DIR = LIB_DIR.joinpath("CMRSegment", "resource")
