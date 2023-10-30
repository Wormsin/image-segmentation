# image-segmentation

### description
**auto_seg.py** creates the dataset from the video or set of images for training yolo segmentation model.

**clustering_alg.py** has segmentation machine learning algorithms. 

**run_yolo.py** addes a new class to dataset-seg.yaml and trains yolov5 on data that were made by auto_seg.py for this class.

### run auto_seg.py
To enable augmentation add ```--aug```

To make images from a video add ```--video VIDEO_PATH```

To create a specific folder for the video frames add ```--source FOLDER```, by default ```source='images'```

To change the fps add ```--fps NUMBER```, by default ```fps=10```

```ruby
python auto_seg.py --source YOUR_FOLDER_WITH_IMAGES --aug
```
### run run_yolo.py

```ruby
python run_yolo.py
>> enter the class name: 
```
