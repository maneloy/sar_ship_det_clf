# sar_ship_det_clf
Project consisting on the detection and classification of ships in SAR images, as well as the creation of a dataset to that end.

To create image:

docker build --no-cache -t ship-api .   

To run image:

docker run --rm -p 8000:8000 ship-api