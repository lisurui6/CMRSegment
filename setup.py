from setuptools import setup, find_namespace_packages


with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="CMRSegment",
    use_scm_version=True,
    author="Surui Li",
    author_email="lisurui6@gmail.com",
    description="Segment 3D cardiac MRI cine images, and track their motions.",
    package_dir={"": "lib"},
    packages=find_namespace_packages(where="lib"),
    setup_requires=["setuptools >= 40.0.0"],
    install_requires=install_requires,
    package_data={"": ["*.conf"]},
    entry_points={},
)
