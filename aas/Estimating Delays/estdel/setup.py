from setuptools import setup

setup(name='estdel',
      version='0.0.1',
      description='Estimate interferometer anteanna delays',
      url='https://github.com/andrewasheridan/hera_sandbox/tree/master/aas/Estimating%20Delays/estdel',
      author='Andrew Sheridan',
      author_email='sheridan@berkeley.edu',
      license='MIT',
      package_dir = {'estdel' : 'estdel'},
      packages=['estdel'],
      install_requires=[
          'numpy>=1.2',
          'tensorflow>=1.8.0',
      ],
      zip_safe=False,
      include_package_data=True)

# AIPY/BLOB/V3 SETUP.PY from import to the function