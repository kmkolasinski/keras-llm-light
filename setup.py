from setuptools import find_packages, setup

from keras_llm_light.version import version

setup(
    name="keras_llm_light",
    version=version,
    description="",
    url="https://github.com/kmkolasinski/keras-llm-light",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=False,
    zip_safe=False,
    install_requires=[],
)