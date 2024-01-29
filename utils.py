import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib
from io import BytesIO
import base64

def draw_original(img):
    # Save the image with bounding boxes to a BytesIO buffer
    buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    plt.imsave(buffer, img, format='png')
    buffer.seek(0)

    # Encode the image as base64 for sending in the response
    image_encoded = base64.b64encode(buffer.read()).decode('utf-8')

    # Close the figure to release resources
    plt.close()
    return image_encoded

def draw_bb(img, image_name, results):
    # Ref: https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
    matplotlib.use("agg")
    
    # Create figure and axes
    _, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    confidences = results[0].boxes.conf
    classes_index = results[0].boxes.cls
    classes = [results[0].names[int(idx)] for idx in classes_index]
    xywhn = results[0].boxes.xywhn
    preprocess = results[0].speed["preprocess"]
    inference = results[0].speed["inference"]
    postprocess = results[0].speed["postprocess"]

    for cf, cl, xywhn in zip(confidences, classes, xywhn):
        x, y, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]
        confidence = cf
        class_name = cl

        # Convert to absolute coordinates
        x = int((x - w / 2) * img.shape[1])
        y = int((y - h / 2) * img.shape[0])
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])

        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Display class name and confidence
        plt.text(x, y - 5, f'{class_name} ({confidence:.2f})', color='r')

    title = "{} | {} objects\nspeed per image\npreprocess: {} ms\ninference: {} ms\npostprocess: {} ms".format(
        image_name, len(classes),
        preprocess, inference, postprocess
    )
    title = "{} | {} objects".format(image_name, len(classes))
    plt.title(title)
    
    # Save the image with bounding boxes to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image as base64 for sending in the response
    image_with_bb_encoded = base64.b64encode(buffer.read()).decode('utf-8')

    # Close the figure to release resources
    plt.close()
    
    return image_with_bb_encoded