This folder contains the results of running the minimum samples finder found in the data/tools folder ("minimumSampleRateFinder.py") over a natural envrionment and a lab environment.


Commands run to generate per-scene frequency maps:

python3 varianceMapper.py -f ../../data/ofstad-arena/ofstad-arena-single-compound-eye.gltf -c 0 4.795 0 -s 25 -n 100 --standard-deviation -m max
python3 varianceMapper.py -f ../../data/ofstad-arena/ofstad-arena-single-compound-eye.gltf -c 0 4.795 0 -s 25 -n 400 --standard-deviation -m mean


python3 varianceMapper.py -f "../../data/natural-standin-sky.gltf" -c -23.029 26.526 0 -s 1098 -n 100 --standard-deviation -m max
python3 varianceMapper.py -f "../../data/natural-standin-sky.gltf" -c -23.029 26.526 0 -s 1098 -n 400 --standard-deviation -m mean


Commands run to perform minimum sampling search:
python3 minimumSampleRateFinder.py -f "../../data/ofstad-arena/ofstad-arena-single-compound-eye.gltf" -s cylinder -c 0 0.3 0 12.5 9.19 -p 0.01

python3 minimumSampleRateFinder.py -f "../../data/natural-standin-sky.gltf" -s box -b -572.029 26.5 -549 525.971 26.6 549 -p 0.01
