from setuptools import setup

setup(
    name="intercycle_sim",
    version="0.1.0",
    packages=["intercycle_sim"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/intercycle_sim"]),
                ("share/intercycle_sim", ["package.xml"]),
                ("share/intercycle_sim/config", ["config/nav2_gate.yaml"])],
    install_requires=["setuptools"],
    license="MIT",
    entry_points={"console_scripts": [
        "sim_bridge = intercycle_sim.ros_bridge_node:main",
        "send_goal = intercycle_sim.send_goal:main",
    ]},
)
