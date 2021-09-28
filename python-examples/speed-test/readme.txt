This folder contains the scripts and results for and of running speed tests on
various GPUs.

Speed tests were run using 1000 samples for averaging, and were rendered from 1 to 5000 samples per ommatidium:
python3 speedTest.py --lib ../../build/make/lib/libEyeRenderer3.so -f ../scenes/ofstad-arena-single-compound-eye.gltf -o ofstad -s 1000 -m 5000

And then the generated plotted using plotter.py
