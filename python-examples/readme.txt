Here you'll find some example usages of the eye renderer using it's Python bindings.

`primary-example.py` provides a good overview of the process of using the eye renderer, while each subfolder represents a specific example that relates to the figures found in this paper: https://elifesciences.org/articles/73893
Please note that due to the current unavailability of the 'natural' environment (also referenced here as the 'rothamsted' environment), a stand-in environment is currently in use, accessed via 'compound-ray/data/natural-standin-sky.gltf`

Note that all of these python examples require this folder (the `python-examples` folder) to be source-able, so please add it to your python path using `$ export PYTHONPATH="/path/to/python-examples/"`
In order to use the eyeRendererHelperFunctions module, you may beed to insert the `python-examples` folder into your system path on Python launch before importing the eye tools using a line like so:
```
sys.path.insert(1, '/path/to/python-path')
import eyeRendererHelperFunctions as eyeTools
```
