from setuptools import setup, find_namespace_packages


setup(
    name="CMRSegment",
    use_scm_version=True,
    author="Surui Li",
    author_email="lisurui6@gmail.com",
    description="Segment 3D cardiac MRI cine images, and track their motions.",
    package_dir={"": "lib"},
    packages=find_namespace_packages(where="lib"),
    setup_requires=["setuptools >= 40.0.0"],
    package_data={"": ["*.conf", "*.txt"]},
    entry_points={
        "console_scripts": [
            "cmrtk-pipeline = CMRSegment.pipeline.cli:main",
            "cmrtk-torch-segmentor = CMRSegment.segmentor.torch.cli:main",
        ]
    },
    include_package_data=True,
)
