from PIL import Image
import os

# filepath = "C:\\Users\\ninanpyo\\Pictures\\Saved Pictures\\40167808_10213079885337252_7529443021281034240_n.jpg"
# # dir_fp = "C:\\Users\\ninanpyo\\Pictures\\Saved Pictures\\"
# dir_fp = "C:\\Users\\ninanpyo\\Desktop\\datasets\\Training\\00000"


def get_image(img_filepath):
    img = Image.open(img_filepath)
    pixels = list(img.getdata())
    return pixels, img.size


def get_image_dir(dir_filepath):
    files = []
    for file in os.listdir(dir_filepath):
        try:
            files.append(get_image(os.path.join(dir_filepath, file)))
        except:
            pass
    return files


# get_image_dir(dir_fp)
