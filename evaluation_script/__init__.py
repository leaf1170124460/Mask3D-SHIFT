"""
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.

"""

import os
import subprocess
import sys
from pathlib import Path


def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_local_package(folder_name):
    # Install a local python package

    # Args:
    #     folder_name ([str]): name of the folder placed in evaluation_script/

    subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            os.path.join(str(Path(__file__).parent.absolute()) + folder_name),
        ]
    )


# subprocess.check_call(["git", "--version"])
# subprocess.check_call(["git", "lfs", "--version"])
# subprocess.check_call(["git", "lfs", "pull"])

install("numpy")
install("tqdm")
install("git+https://github.com/scalabel/scalabel.git")
install("nuscenes-devkit==1.1.10")
install("pyquaternion==0.9.9")
install("Pillow")

print("============")
print("Install done")
print("============")


from .main import evaluate
