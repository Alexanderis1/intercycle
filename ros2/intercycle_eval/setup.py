from setuptools import setup

setup(
    name="intercycle_eval",
    version="0.1.0",
    packages=["intercycle_eval"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/intercycle_eval"]),
                ("share/intercycle_eval", ["package.xml"])],
    install_requires=["setuptools"],
    license="MIT",
)
