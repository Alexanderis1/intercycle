from setuptools import setup

setup(
    name="intercycle_nav",
    version="0.1.0",
    packages=["intercycle_nav"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/intercycle_nav"]),
                ("share/intercycle_nav", ["package.xml"]),
                ("share/intercycle_nav/config", ["config/nav2.yaml"]),
                ("share/intercycle_nav/launch", ["launch/nav.launch.py", "launch/observer.launch.py"])],
    install_requires=["setuptools"],
    license="MIT",
    entry_points={"console_scripts": [
        "observer_bridge = intercycle_nav.observer_bridge:main",
        "timing_planner = intercycle_nav.timing_planner:main",
        "camera_stub = intercycle_nav.camera_stub:main",
        "devfeed = intercycle_nav.devfeed:main",
    ]},
)
