import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VoidGaze",
    version="0.0.1",
    author="Bojia Mao",
    author_email="bojia.mao16@imperial.ac.uk",
    description="MSc Computing Individual Project - Gaze tracking by head tracking in virtual reality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LUI-TECH/VoidGaze",
    project_urls={
        "Bug Tracker": "https://github.com/LUI-TECH/VoidGaze/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)