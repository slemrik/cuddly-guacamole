from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['cuddlyguacamole.test.run_tests']
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

description = '''
Simulator of system of charged particles...
'''

setup(
    cmdclass={'test': PyTest},
    name='cuddlyguacamole',
    version='0.1.0',
    author='Henrik Gjoertz et al.',
    author_email='henrik.gjoertz@fu-berlin.de',
    url='https://github.com/slemrik/cuddly-guacamole',
    description=description,
    packages=['cuddlyguacamole', 'cuddlyguacamole.test'],
    setup_requires=['pytest-runner',],
    install_requires=['numpy'],
    tests_require=['pytest'],
    zip_safe=False)
