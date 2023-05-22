from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list of requirements
    '''

    H = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if H in requirements:
            requirements.remove(H)
    
    return requirements




setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Sujai Sheell Chaudhary',
    author_email = 'sujaichaudhary98@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)
