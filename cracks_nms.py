from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img,img_to_array
#from mrcnn.visualize import display_instances
from matplotlib.patches import Rectangle
from mrcnn.model import MaskRCNN
import glob
import cv2
from mrcnn.config import Config
import numpy as np
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import expand_dims
from mrcnn.utils import compute_ap
from numpy import mean
from display import display_instances,apply_mask


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "cracks_cfg"
	# number of classes (background + crack)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/cracks_cfg20210427T1057/mask_rcnn_cracks_cfg_0003.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        #image1 = dataset.load_image(i)
        mask, class_ids = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.figure(figsize=(39,29))
        #pyplot.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        figure = pyplot.gcf()
        figure.set_size_inches(12, 15)
        pyplot.savefig('./output/actual{}.jpg'.format(i),dpi=100)
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.5,aspect='auto')
        # get the context for drawing boxes
        #pyplot.figure(figsize=(39,29))
        #pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='blue')
            # draw the box
            ax.add_patch(rect)
    # show the figure
        #figure = pyplot.gcf()
        #figure.set_size_inches(50, 40)
        #pyplot.savefig('./output/predicted{}.jpg'.format(i),dpi=100)
        #con = concatenate((image,image1),axis=1)
        #cv2.imwrite('/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/output/crack{}'.format(i),con)
        #pyplot.savefig('./output/predicted{}.jpg'.format(i),dpi=100)
        figure = pyplot.gcf()
        figure.set_size_inches(12, 15)
        pyplot.savefig('./output/predicted{}.jpg'.format(i),dpi=100)
        
    #pyplot.show()

# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)

new_set = CracksDataset()
new_set.load_dataset('input')
new_set.prepare()
print('Test: %d' % len(test_set.image_ids))

plot_actual_vs_predicted('300.jpeg', model, cfg)

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "cracks_cfg"
	# number of classes (background + crack)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_cracks_cfg_0003.h5', by_name=True)

img = load_img('/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/input/images/301.jpeg')
img = img_to_array(img)

def draw_image_with_boxes(filename,boxes_list):
    data = pyplot.imread(filename)
    pyplot.figure(figsize=(15,20))
    pyplot.imshow(data)
    ax = pyplot.gca()
    for box in boxes_list:
        y1,x1,y2,x2 = box
        width,height = x2-x1,y2-y1
        rect = Rectangle((x1,y1),width,height,fill=False,color='blue',lw=2)
        ax.add_patch(rect)
    pyplot.show()

class_names = ['BG','Crack']
path = '/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/input/images/*.*'
for num,InputImage in enumerate(glob.glob(path)):
    img = load_img(InputImage)
    img1 = img_to_array(img)
#    cv2.imwrite('/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/output/actual{}.jpg'.format(num),img1)
    results = model.detect([img1],verbose=0)
    draw_image_with_boxes(InputImage,results[0]['rois'])
    r = results[0]
    display_instances(img1,r['rois'],r['masks'],r['class_ids'],class_names,r['scores'])
#    cv2.imwrite('/home/smohammad/Projects/Custom-Cracks/Label_Crack+Nocrack/output/predicted{}.jpg'.format(num),img1)

def box_iou(boxes1, boxes2):
    """Compute IOU between two sets of boxes of shape (N,4) and (M,4)."""
    # Compute box areas
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    unioun = area1[:, None] + area2 - inter
    return inter / unioun

def nms(boxes, scores, iou_threshold):
    # sorting scores by the descending order and return their indices
    B = scores.argsort()[::-1]
    keep = []  # boxes indices that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)

