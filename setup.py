from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='Functions for abstaining',
          url='NA',
          download_url='NA',
          version='0.1',
          packages=['abstention'],
          setup_requires=[],
          install_requires=['numpy>=1.9', 'keras>=1.0'],
          scripts=[],
          name='abstention')
