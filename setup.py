from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='Functions for abstention, calibration and label shift domain adaptation',
          url='https://github.com/kundajelab/abstention',
          version='0.1.3.1',
          packages=['abstention'],
          setup_requires=[],
          install_requires=['numpy>=1.9',
                            'scikit-learn>=0.20.0',
                            'scipy>=1.1.0'],
          scripts=[],
          name='abstention')
