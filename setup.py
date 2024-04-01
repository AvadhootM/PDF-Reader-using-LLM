from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements] ## Replaces \n by blank when the line is read from requriements

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT) ## '-e .' from requriements.txt, should not come in the result of this file

setup(
    name='PDF-Reader-using-LLM',
    version='0.0.1',
    author='Avadhoot',
    author_email='avadhootmurugkar3@gamil.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)