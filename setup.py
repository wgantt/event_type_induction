from setuptools import find_packages, setup

setup(
    name="event_type_induction",
    version="0.1",
    description="Event type induction and predction on Universal Decompositional Semantics",
    author="Will Gantt, Aaron Steven White",
    author_email="wgantt@cs.rochester.edu, aaron.white@rochester.edu",
    license="MIT",
    packages=find_packages(),
    package_dir={"event_type_induction": "event_type_induction"},
    install_requires=[
        "decomp==2.0.0a1",
        "decorator==4.4.*",
        "networkx==2.5.*",
        "numpy==1.17.*",
        "overrides==3.1.*",
        "setuptools==49.6.*",
        "torch==0.4.*",
        "typing==3.7.*",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    zip_safe=False,
)
