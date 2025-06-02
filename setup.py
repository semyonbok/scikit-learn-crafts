import re
from setuptools import setup, find_packages
from pathlib import Path


def read_version():
    pattern_ = r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]'
    with open("src/crafts/version.py", "r") as f:
        return re.search(pattern_, f.read()).group(1)


# Read the long description from README.md if present
ROOT_PATH = Path(__file__).parent.resolve()
try:
    with open(ROOT_PATH / "README.md", encoding="utf-8") as f:
        long_desc = f.read()
except FileNotFoundError:
    long_desc = ""

setup(
    name="scikit-learn-crafts",
    version=read_version(),
    description="Custom scikit-learn objects",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="semyonbok",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "pandas", "scikit-learn", "joblib"],
)
