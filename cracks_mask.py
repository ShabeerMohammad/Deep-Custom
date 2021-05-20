# detect kangaroos in photos with mask rcnn model
import cv2
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
from numpy import concatenate
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import time
# class that defines and loads the kangaroo dataset
class CracksDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Cracks")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 175:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 175:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('Cracks'))
		return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# load the train dataset
train_set = CracksDataset()
train_set.load_dataset('Mask_Cracks', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# load the test dataset
test_set = CracksDataset()
test_set.load_dataset('Mask_Cracks', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# define a configuration for the model
class CracksConfig(Config):
	# define the name of the configuration
	NAME = "Cracks_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 174

# prepare config
config = CracksConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
print('Model Training Started!')
start_time = time.time()
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
end_time = time.time()
elapsed_time = end_time - start_time
print('Done!! Took {} seconds'.format(elapsed_time))
