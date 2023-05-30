from setuptools import find_packages, setup

from typing import List

hyphen_e = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
    returns the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", " ") for req in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)

            return requirements



setup(
    name = 'Housing Prices Prediction',
    version = '0.1',
    auther = 'Sachin Ubale',
    auther_email = 'ubalesachin22@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )