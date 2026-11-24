# Build the optional native adapter with the same CMake targets as the C++ library.
import os
from pathlib import Path
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        root = Path(__file__).resolve().parent
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / ext.name
        subprocess.run([
            "cmake", "-S", str(root), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DNDP_BUILD_PYTHON=ON", "-DBUILD_TESTING=OFF",
            "-DNDP_PYTHON_TESTING=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}", f"-DNDP_PYTHON_OUTPUT_DIR={output}",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "ndp_python",
                        "-j", os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2")], check=True)


setup(ext_modules=[Extension("ndp._C", sources=[])], cmdclass={"build_ext": CMakeBuild})
