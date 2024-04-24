from setuptools import setup

setup(  
    name= 'VirtualTrailRoom', 
    version='0.0.1', 
    description='The package is a virtual trail room for the customers to test clothes virtually.', 
    py_modules=["maths"],
    package_dir={'': 'src'},
    install_requires = ["blessings ~= 1.7"],
    extras_require={
        "dev": [
            "pytest>=3.7",
        ],
    },
)

