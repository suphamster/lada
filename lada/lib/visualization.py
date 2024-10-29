import cv2


def overlay_mask(frame, mask):
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #overlay[:,:,1][overlay[:,:,1] > 0] = 30 # add some color
    output = frame.copy()
    alpha = 0.1
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def overlay_mask_boundary(frame, mask, color=(0, 255, 0)):
    output = frame.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, color, 2)
    return output

def overlay_edges(frame, edges, color=(0, 255, 0)):
    output = frame.copy()
    output[edges.squeeze() > 0] = color
    return output

def overlay(background, overlay):
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    output = background.copy()
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def draw_box(img, box, color=(255, 0, 0), thickness = 2):
    start_point, end_point = (box[1], box[0]), (box[3], box[2])
    cv2.rectangle(img, start_point, end_point, color, thickness)

def draw_text(text, position, output, font_scale=0.5):
    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                cv2.LINE_AA)