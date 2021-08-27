import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img

def check_paths(src, target):
    # Fix the handling og the file names => [:-16] was case specific; it does not work
    assert len(src) == len(target), print("Inputs 'src' and 'target' must have the same length")
    
    check=np.zeros(len(src))
    for i, (f1, f2) in enumerate(zip(src, target)):
        check[i] = os.path.split(f1)[1][:-16] not in f2
    print(f"Number of source file without corresponding target file: {check.sum()}")
    return(check)

def get_paths(src_dir, target_dir, source_type, target_type, verbose=True):
    
    set_type = ['train', 'val', 'test']
    
    src_paths = dict()
    target_paths = dict()
    
    for s in set_type:
        src_paths[s] = sorted(glob.glob(os.path.join(src_dir,'*',s,'*',source_type)))
        target_paths[s] = sorted(glob.glob(os.path.join(target_dir,'*',s,'*',target_type)))
        if verbose == True:
            print(f"\nNumber of {s} files:")
            print(f"    in the source directory: {len(src_paths[s])}")
            print(f"    in the target directory: {len(target_paths[s])}")
            #check_paths(src_paths[s], target_paths[s])

    return(src_paths, target_paths)

def resize(src_paths, target_paths, height,  width):

    l = len(src_paths)
    for i, path in enumerate(src_paths):
        if i%100 == 0:
            print(f"Source reduction: {i}/{l}")
        img = load_img(path, color_mode='rgb', target_size=(height, width))
        resize_path = path[:-4] + '_reduced.png'
        img.save(resize_path)

    l = len(target_paths)    
    for i, path in enumerate(target_paths):
        if i%100 == 0:
            print(f"Target reduction: {i}/{l}")
        img = load_img(path, color_mode='grayscale', target_size=(height, width))
        resize_path = path[:-4] + '_reduced.png'
        img.save(resize_path)

def remove(src_paths, target_paths, suffix):

    paths = [p for p in src_paths if p.endswith(suffix)]
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    paths = [p for p in target_paths if p.endswith(suffix)]
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def convert_id2cat(target_paths, id2cat):

    categories = set(id2cat.values())
        
    for path in target_paths:

        img = load_img(path, color_mode="grayscale")

        img_array = np.array(img)
        img_array = img_array.flatten()

        for i in categories:
            test_elements = [id for id in id2cat if id2cat[id]==i]
            idx = np.isin(img_array,test_elements)
            img_array[idx] = i

        img_array = img_array.reshape(img.height, img.width, -1)
        convert_path = path[:-4] + '_cats.png'

        save_img(convert_path, img_array, scale=False)

def serialize_image(image):
    import json
    import numpy as np
    import io
    img = np.array(image)
    memfile = io.BytesIO()
    np.save(memfile, img)
    memfile.seek(0)
    data_serialized = json.dumps({'data':memfile.read().decode('latin-1')})
    return(data_serialized)


def deserialize_image(data_serialized):
    import json
    from PIL import Image
    import io
    memfile = io.BytesIO()
    memfile.write(json.loads(data_serialized)['data'].encode('latin-1'))
    memfile.seek(0)
    img = np.load(memfile)
    image = Image.fromarray(img)
    return(image)

def deserialize_mask(data_serialized):
    import json
    import io
    memfile = io.BytesIO()
    memfile.write(json.loads(data_serialized)['data'].encode('latin-1'))
    memfile.seek(0)
    msk = np.load(memfile)
    return(msk)