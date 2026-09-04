from setuptools import setup

setup(
    name="streamllm",
    version="0.1.1",
    packages=["streamllm"],
    entry_points={
        "console_scripts": [
            "streamllm = streamllm.cli:main",
        ],
    },
)
