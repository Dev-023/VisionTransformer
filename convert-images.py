from PIL import Image

path = "dataset/ImageNetSelected/n01735189_11267"
im = Image.open(path + ".JPEG")
im.save(path + ".ppm")

