import cv2
import numpy as np

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
def GetLocation(move_type, env, current_frame, predictor):
    shp = current_frame.shape
    img = cv2.resize(np.array(current_frame), (int(shp[0]/4), int(shp[1]/4)))
    outputs = predictor(img)

    predictions = outputs["instances"].pred_boxes.tensor.numpy()
    results = []

    if len(predictions) == 0:
        print("I see nothing")
    for pred in predictions[:5]:
        y = int((pred[0]+pred[2])*1.5)
        x = int((pred[1]+pred[3])*1.5)
        results.append({'coordinate' : (x+100, y), 'move_type' : move_type})
        results.append({'coordinate' : (x-100, y), 'move_type' : move_type})

    return results

