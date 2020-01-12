import six

def get_labels(boxes , classes , scores , category_index , min_score_thresh = .5):

    max_boxes_to_draw = boxes.shape[0]
    detected_labels = []

    for i in range(min(max_boxes_to_draw , boxes.shape[0])):

        if scores[i] < min_score_thresh:
            continue


        if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]["name"]
            score = int(100 * scores[i])
            print("{} => {}".format(class_name , score))
            detected_labels.append()

    return detected_labels
