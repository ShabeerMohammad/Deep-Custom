from PIL import Image
import numpy as np
from matplotlib import pyplot
from cv2 import findContours
from matplotlib.patches import Polygon
from matplotlib import patches
import random

def display_instances(image, boxes, masks, class_ids, class_names,
                  scores, title="",
                  figsize=(16, 16), ax=None,
                  show_mask=True, show_bbox=True,
                  colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
        """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = pyplot.subplots(1, figsize=figsize)
        auto_show = True
    
    # Generate random colors
    colors = colors or random_color(N)
    
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
    
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                      linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)
    
        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
    
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
    
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = findContours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    img1 = Image.fromarray(masked_image.astype(np.uint8), 'RGB')
    img1.save('my.png')
    img1.show()
    
    if auto_show:
        pyplot.show()
        
        
def apply_mask(image, mask, invert=False):
    image.alpha_channel = True
    if invert:
        mask.negate()
    with Image(width=image.width, height=image.height,
               #background=Color("transparent")
               ) as alpha_image:
        alpha_image.composite_channel(
            "alpha",
            mask,
            "copy_opacity",
            0, 0)
        image.composite_channel(
            "alpha",
            alpha_image,
            "multiply",
            0, 0)
        
def random_color(i):
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(i))