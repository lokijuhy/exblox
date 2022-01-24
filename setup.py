from setuptools import find_packages, setup


# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
      name='exblox',
      version='0.0.1',
      python_requires='>=3.7.3',
      description='',
      url='https://github.com/lokijuhy/exblox',
      packages=find_packages(exclude=["tests"]),
      install_requires=requires,
      extras_requires={
            'dev': ['pytest'],
      },
      zip_safe=False)
