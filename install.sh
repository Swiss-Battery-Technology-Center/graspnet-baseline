cd grapnet-baseline
pip install -r requirements.txt
cd pointnet2
python setup.py install --user
cd ..
cd knn
python setup.py install --user
cd ..
cd graspnetAPI
pip install .