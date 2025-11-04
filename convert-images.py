from PIL import Image

path = "dataset/ImageNetSelected/n01687978_10071"
im = Image.open(path + ".JPEG")
im.save(path + ".ppm")

