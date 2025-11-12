import pdb

import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}


def remove_small(binary_mask, min_area=10000):
    binary_mask = binary_mask.astype(np.uint8)

    label_image = label(binary_mask)
    regions = regionprops(label_image)
    small_white_regions = [region for region in regions if region.area < min_area]

    for region in small_white_regions:
        min_row, min_col, max_row, max_col = region.bbox
        binary_mask[min_row:max_row, min_col:max_col] = 0
    return binary_mask

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def elongation_mask(mask_array, stretch_factor=None):
    mask_array = np.array(mask_array)
    rows = np.any(mask_array, axis=1)
    upper, lower = np.where(rows)[0][[0, -1]]
    # 计算新的长度
    if not stretch_factor:
        stretch_factor = 0.95-np.random.rand(1)[0]*0.25
        # print(stretch_factor)
        # radio = 1-lower/mask_array.shape[0]
        # if radio<=0.1:
        #     stretch_factor = np.random.rand(1)[0]*0.1+1.0
        # elif 0.1<radio<0.2:
        #     stretch_factor = np.random.rand(1)[0]*0.2+1.0
        # else:
        #     stretch_factor = np.random.rand(1)[0]*0.35+1.0
    stretch_length = min(int((lower - upper + 1) * stretch_factor), mask_array.shape[0]-upper)

    # 拉伸mask区域
    mask_region = mask_array[upper:lower+1, :]  # 取mask区域
    stretched_mask_region = cv2.resize(mask_region, (mask_array.shape[1], stretch_length), interpolation=cv2.INTER_NEAREST)

    # 如果需要，压缩mask区域下方的内容
    remaining_height = mask_array.shape[0] - (lower + 1)
    compressed_height = mask_array.shape[0] - stretch_length - upper
    if remaining_height > 0 and compressed_height > 0:
        remaining_region = mask_array[lower+1:, :]
        compressed_remaining_region = cv2.resize(remaining_region, (mask_array.shape[1], compressed_height), interpolation=cv2.INTER_AREA)
    else:
        compressed_remaining_region = np.array([])

    # 创建新的mask图像
    new_mask = np.zeros_like(mask_array)
    new_mask[:upper] = mask_array[:upper]  # 上边界保持不变
    new_mask[upper:upper + stretch_length] = stretched_mask_region
    if compressed_remaining_region.size > 0:
        new_mask[upper + stretch_length:] = compressed_remaining_region
    return new_mask

def get_img_agnostic_upper_rectangle(im_parse, pose_data):
    pose_data = pose_data[0]
    pose_data[pose_data<0]=0
    pose_data[:,0]*=im_parse.shape[1]
    pose_data[:,1]*=im_parse.shape[0]

    foot = pose_data[18:24]

    faces = pose_data[24:92]

    hands1 = pose_data[92:113]
    hands2 = pose_data[113:]
    body = pose_data[:18]
    parse_array = np.array(im_parse)
    parse_upper_all = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 7).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32)
                    )
    parse_upper = ((parse_array == 4).astype(np.float32) +
                     (parse_array == 7).astype(np.float32)
                    )
    parse_head = ((parse_array == 3).astype(np.float32) +
                    (parse_array == 1).astype(np.float32) + (parse_array == 11).astype(np.float32))
    parse_fixed = (parse_array == 16).astype(np.float32)


    agnostic = Image.new(mode='L',size=(parse_array.shape[1], parse_array.shape[0]), color=0)
    img_black = Image.new(mode='L',size=(im_parse.shape[1], im_parse.shape[0]),color=0)

    gray_img = Image.new('L', size=(im_parse.shape[1], im_parse.shape[0]), color=128)
    agnostic_draw = ImageDraw.Draw(agnostic)

    parse_upper = np.uint8(parse_upper*255)
    parse_upper_all = np.uint8(parse_upper_all*255)
    # 查找轮廓
    contours_all, _ = cv2.findContours(parse_upper_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(parse_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    cloth_exist = False
    try:
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 100:
                x, y, w, h = cv2.boundingRect(contour)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                cloth_exist = True
        _x1, _y1, _x2, _y2 = min_x, min_y, max_x, max_y
    except:
        cloth_exist = False

    x1 = [body[i, 0] for i in [2, 3, 4] if body[i, 0] != 0]+\
        [hands2[i, 0] for i in [5,9,13] if hands2[i, 0] != 0]+\
        [hands1[i, 0] for i in [5,9,13] if hands1[i, 0] != 0]
    if x1:
        x1 = np.min(x1)
    else:
        x1 = 0
    y1 = [body[i, 1] for i in [2, 5] if body[i, 1] != 0]+\
          [hands2[i, 1] for i in [5,9,13] if hands2[i, 1] != 0]+\
            [hands1[i, 1] for i in [5,9,13] if hands1[i, 1] != 0]
    if y1:
        y1 = np.min(y1)
    else:
        y1 = 0
    x2 = [body[i, 0] for i in [5, 6, 7] if body[i, 0] != 0]+\
        [hands1[i, 0] for i in [5,9,13] if hands1[i, 0] != 0]+\
        [hands2[i, 0] for i in [5,9,13] if hands2[i, 0] != 0]
    if x2:
        x2 = np.max(x2)
    else:
        x2 = parse_array.shape[1]
    y2 = [body[i, 1] for i in [8, 11] if body[i, 1] != 0]+\
        [hands2[i, 1] for i in [5,9,13] if hands2[i, 1] != 0]+\
        [hands1[i, 1] for i in [5,9,13] if hands1[i, 1] != 0]
    if y2:
        y2 = np.max(y2)
    else:
        y2 = parse_array.shape[0]

    pad_y1 = 20
    pad_y2 = 10
    pad_x1 = pad_x2 = 30

    if cloth_exist:
        x1 = min(x1, _x1)
        x2 = max(x2, _x2)
        y1 = min(y1, _y1)
        if y2>_y2:
            pad_y2 = 0
        y2 = max(y2, _y2)

    y_face = [faces[i, 1] for i in [5, 11] if faces[i, 1] != 0]

    if y_face:
        y_face = np.mean(y_face)
        if y_face<y1:
            pad_y1 = 0
        y1 = min(y1, y_face)



    agnostic_draw.rectangle((max(0, x1-pad_x1), max(0, y1-pad_y1), min(x2+pad_x2, parse_array.shape[1]), min(y2+pad_y2, parse_array.shape[0])), 'gray', 'gray')
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img_black, None, Image.fromarray(np.uint8(parse_fixed * 255), 'L'))

    mask_gray = agnostic.copy()
    mask = agnostic.point(lambda p: p == 128 and 255)
  
    return mask, mask_gray


def get_mask_location_v1(category, model_parse: Image.Image, pose_data: np.array, width=384,height=512, densepose=None, depthmap=None):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)
    # import pdb;pdb.set_trace()
    # Load pose points
    pose_data[:,0]*=float(width)
    pose_data[:,1]*=float(height)

    if category== "upper_body":
        mask, mask_gray = get_img_agnostic_upper_rectangle(parse_array, pose_data)
        return mask, mask_gray
    else:
        parse_head = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 3).astype(np.float32) + \
                    (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms_left = (parse_array == 14).astype(np.float32)
        arms_right = (parse_array == 15).astype(np.float32)
        arms = arms_left + arms_right

        if category == 'dresses':
            parse_mask = (parse_array == 7).astype(np.float32) + \
                        (parse_array == 4).astype(np.float32) + \
                        (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif category == 'upper_body':
            parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
            parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                            (parse_array == label_map["pants"]).astype(np.float32)
            parser_mask_fixed += parser_mask_fixed_lower_cloth
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif category == 'lower_body':
            parse_mask = (parse_array == 6).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32) + \
                        (parse_array == 5).astype(np.float32)
            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                (parse_array == 14).astype(np.float32) + \
                                (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        else:
            raise NotImplementedError

        im_arms_left = Image.new('L', (width, height))
        im_arms_right = Image.new('L', (width, height))
        arms_draw_left = ImageDraw.Draw(im_arms_left)
        arms_draw_right = ImageDraw.Draw(im_arms_right)
        if category == 'dresses' or category == 'upper_body':
            shoulder_right = pose_data[2][:2]
            shoulder_left = pose_data[5][:2]
            elbow_right = pose_data[3][:2]
            elbow_left = pose_data[6][:2]
            wrist_right = pose_data[4][:2]
            wrist_left = pose_data[7][:2]
            ARM_LINE_WIDTH = 110
            size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
            size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                        shoulder_right[1] + ARM_LINE_WIDTH // 2]
            

            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                im_arms_right = arms_right
            else:
                wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
                arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                im_arms_left = arms_left
            else:
                wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
                arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
            hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
            parser_mask_fixed += hands_left + hands_right

        parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        if category == 'dresses' or category == 'upper_body':
            neck_mask = (parse_array == 18).astype(np.float32)
            neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
            neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
            parse_mask = np.logical_or(parse_mask, neck_mask)
            # arm_mask = np.logical_or(im_arms_left, im_arms_right)
            arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
            parse_mask += np.logical_or(parse_mask, arm_mask)

        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        inpaint_mask = 1 - parse_mask_total
        img = np.where(inpaint_mask, 255, 0)
        dst = hole_fill(img.astype(np.uint8))
        dst = refine_mask(dst)
        inpaint_mask = dst / 255 * 1
        mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 128)
        return mask, mask_gray
