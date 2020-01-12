import six

def get_labels(boxes , classes , scores , category_index):

    max_boxes_to_draw = boxes.shape[0]
    detected_labels = []

    for i in range(min(max_boxes_to_draw , boxes.shape[0])):
        if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]["name"]
            detected_labels.append(class_name)

    return detected_labels
