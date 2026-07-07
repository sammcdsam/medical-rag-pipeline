"""pytest bootstrap — put the project root on sys.path so tests can `import access`
etc. from the flat module layout without installing the project as a package."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
