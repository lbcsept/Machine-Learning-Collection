


def input_shape_from_image_shape(image_shape=(448, 448, 3), batch_size=2):
    """Return shape (batch_size, c, h, w) from image shape (h, w, c) """
    image_shape = list(image_shape)
    input_shape = [batch_size] + [image_shape[-1]] + image_shape[:-1]
    return tuple(input_shape)
