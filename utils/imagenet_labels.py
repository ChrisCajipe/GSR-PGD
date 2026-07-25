DOG_CLASSES = set(range(151,268))

def is_dog(prediction):
    return prediction in DOG_CLASSES