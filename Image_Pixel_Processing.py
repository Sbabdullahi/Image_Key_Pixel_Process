import glob
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# Given a box with pairs of nodes, the function creates edge components as set.
def connected_components(edges):
    neighbors = defaultdict(set)
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen = set()

    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        unseen = set([node])
        next_unseen = unseen.pop
        while unseen:
            node = next_unseen()
            see(node)
            unseen |= neighbors[node] - seen
            yield node

    return (set(component(node)) for node in neighbors if node not in seen)


# The function will generate pixel coordinates concerning the pixel that fulfill the test
def matching_pixels(image, test):

    width, height = image.size
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            if test(pixels[x, y]):
                yield x, y


# The function will generate pairs of neighboring pixel coordinates
def make_edges(coordinates):

    coordinates = set(coordinates)
    for x, y in coordinates:
        if (x - 1, y - 1) in coordinates:
            yield (x, y), (x - 1, y - 1)
        if (x, y - 1) in coordinates:
            yield (x, y), (x, y - 1)
        if (x + 1, y - 1) in coordinates:
            yield (x, y), (x + 1, y - 1)
        if (x - 1, y) in coordinates:
            yield (x, y), (x - 1, y)
        yield (x, y), (x, y)


# The function will yield the bounding box of all coordinates
def boundingbox(coordinates):

    xs, ys = zip(*coordinates)
    return min(xs), min(ys), max(xs), max(ys)


# The function will yield the bounding boxes of all non-consecutive regions
def disjoint_areas(image, test):

    for each in connected_components(make_edges(matching_pixels(image, test))):
        yield boundingbox(each)

# The value settings are declared
def is_black_enough(pixel):
    r, g, b = pixel
    return r < 5 and g < 5 and b < 5


# The testing of the single image is performed here to display the actual key pixels and the non-consecutive pixels
if __name__ == '__main__':

    image = Image.open(r"E:/Arabic_Sign_language_Recognition/ArSL_images/datasets/train/images/1_21_M_ain_0.jpg")
    # image = Image.open('1_21_M_ain_0.jpg')
    draw = ImageDraw.Draw(image)
    for rect in disjoint_areas(image, is_black_enough):
        draw.rectangle(rect, outline=(255, 0, 0))
    image.show()
