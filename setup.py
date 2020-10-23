from setuptools import setup, find_packages


with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="CMRSegment",
    use_scm_version=True,
    author="Surui Li",
    author_email="lisurui6@gmail.com",
    description="Segment 3D cardiac MRI cine images, and track their motions.",
    packages=find_packages(),
    setup_requires=["setuptools >= 40.0.0"],
    install_requires=install_requires,
    package_data={"": ["*.conf"]},
    entry_points={
            'console_scripts': [
                'launch-experiment = CMRSegment.nn.experiment.launcher:main',
            ]
        },
)
