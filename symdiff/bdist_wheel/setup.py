from setuptools import setup, Distribution
from glob import glob
import os
import re

#https://stackoverflow.com/questions/35112511/pip-setup-py-bdist-wheel-no-longer-builds-forced-non-pure-wheels/36886459#36886459
#https://stackoverflow.com/questions/24071491/how-can-i-make-a-python-wheel-from-an-existing-native-library/24793171#24793171
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

def fix_readme(text):
    regex = re.compile('\[([^]]+)\]+\((?!http[s]:\/\/)([^)]+)\)')
    base_url = os.environ.get('README_BASE_URL')
    if base_url:
        if base_url[-1] != '/':
            base_url += '/'
        print("Fixing up base url for long description {0}".format(base_url))
        text = regex.sub(lambda m, urlbase=base_url : r'[{0}]({1})'.format(m.group(1), urlbase + m.group(2)), text)
    return text

with open('README.md', 'r') as ifile:
    text = ifile.read()
    text = fix_readme(text)

setup(
      packages=['symdiff',],
      package_data={
        'symdiff' : ['./*.so', './*.dll', './*.dylib', './*.pyd',],
      },
      distclass=BinaryDistribution,
      data_files = [
        ('symdiff_data',  ['LICENSE', 'NOTICE', 'README.md',]),
        ('symdiff_data/doc', glob('doc/*.pdf')),
        ('symdiff_data/examples', glob('examples/*.*')),
      ],
      long_description = text,
      long_description_content_type="text/markdown",
    )
