from setuptools import setup

setup(
    name='emotion_net',
    version='0.0.1',
    keywords='emotion, valence, arousal, affectNet',
    url='',
    description='landmark, emotion, valence and arousal prediction',
    packages=['emotion_net'],
    install_requires=[
        "scipy >= 1.7.1",
        "numpy >= 1.22.0",
        "onnxruntime >= 1.10.0",
        "opencv_contrib_python >= 4.5.5.64",
    ]
)
