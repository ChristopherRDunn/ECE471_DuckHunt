import pygame, random, cv2
import numpy as np

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
def GetLocation(move_type, env, current_frame, predictor):
    shp = current_frame.shape
    img = cv2.resize(np.array(current_frame), (int(shp[0]/2), int(shp[1]/2)))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = img[:, :, np.newaxis]
    # print("pred")
    outputs = predictor(current_frame)

    # print(outputs["instances"].pred_boxes.tensor)
    predictions = outputs["instances"].pred_boxes.tensor.numpy()
    results = []

    # print(current_frame.shape)
    if len(predictions) == 0:
        print("I see nothing")
    for pred in predictions[:5]:
        y = int((pred[0]+pred[2])/2)
        x = int((pred[1]+pred[3])/2)
        results.append({'coordinate' : (x+160, y), 'move_type' : move_type})
        results.append({'coordinate' : (x-160, y), 'move_type' : move_type})

    # print(results)
    # print("Looking at: ", pygame.mouse.get_pos()[1], pygame.mouse.get_pos()[0])
    return results

