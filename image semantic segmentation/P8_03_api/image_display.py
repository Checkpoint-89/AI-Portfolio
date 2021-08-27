import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def blend_mask2image(image, mask, size, cat2color):
    
    """Quick utility to display a model's prediction."""
    shape = mask.shape
    mask_image = np.zeros(shape+(3,))
    for i in cat2color:
        idx = (mask == i)
        mask_image[idx] = cat2color[i][0]
    mask_image = np.uint8(mask_image)
    mask_image = Image.fromarray(mask_image)
    
    mask_image = mask_image.resize(size)
    image = image.resize(size)
    
    blend = Image.blend(image,mask_image,0.5)

    return(blend)

def get_concat_h(im1, im2):
    dst = Image.new('RGBA', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def add_legend2image(source_img, cat2color, side='left'):

    # create image with size (150,blend width) and black background
    legend_img = Image.new('RGBA', (150,source_img.size[1]), "white")

    # put text on image
    legend_draw = ImageDraw.Draw(legend_img)
    for i in cat2color:
        color = cat2color[i][0]
        cat = cat2color[i][1]
        height_space = source_img.size[1]/len(cat2color)/2
        fontsize = 15
        # legend_draw.text((20, height_space*(i+1)), cat, font=ImageFont.truetype("arial.ttf", fontsize), fill=color)
        legend_draw.text((20, height_space*(i+1)), cat, fill=color)

    # concatenate legend to source image
    if side == 'left':
        dst_img = get_concat_h(legend_img, source_img)
    else:
        dst_img = get_concat_h(source_img, legend_img)

    return(dst_img)