def generate_layer_sizes(dataset_properties, min_layer_size=64, scale_factor=0.5):
    layer_sizes = []
    layer_size = dataset_properties.input_size

    while True:
        layer_size = int(layer_size * scale_factor)
        if layer_size < min_layer_size or layer_size < dataset_properties.num_classes:
            break
        layer_sizes.append(layer_size)

    return layer_sizes
