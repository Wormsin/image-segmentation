# image-segmentation

### description
auto_seg.py creates the dataset for training yolov5 segmentation model. It makes folder seg-dataset with two folders images and labels, images contains a train folder with images that have numerical naming and labels has also the train folder with txt files that contain classes and coordinates of segmetation contours. 
clustering_alg.py has segmentation machine learning algorithms. 

### run
```ruby
python auto_seg.py --source YOUR_FOLDER_WITH_IMAGES
```
