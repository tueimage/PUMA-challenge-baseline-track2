# import setuptools

# if __name__ == "__main__":
#     setuptools.setup()


from setuptools import setup, find_packages

setup(
    name='my_nnunet',
    version='0.1',
    packages=find_packages(exclude=['Input_folder']),  
    # other setup arguments
)