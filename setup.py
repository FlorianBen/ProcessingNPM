#!/usr/bin/env python

from distutils.core import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

name = 'ProcessingNPM'
version = '0.1'
release = '0.1.0'

setup(name=name,
      version=version,
      description='Python script for processing ESS NPM',
      author='Florian Benedetti',
      author_email='florian.benedetti@cea.fr',
      url='https://sedpicc175.extra.cea.fr/gitea/',
      packages=['npm'],
      cmdclass=cmdclass,
      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'release': ('setup.py', release),
              'source_dir': ('setup.py', 'docs/source')}},
      )
