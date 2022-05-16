import datetime
import logging
import os
import ftplib
import numpy
from ftplib import FTP
from pathlib import Path

import cv2
import torch
import itertools


def sgn_final(signum, frame):
    tx = '\n\n Kyle Chuang. \n AUO LT ML5A01 \n Goodbye~😘\n'
    raise StopIteration(f"\033[42;37m{tx}\033[0m")

def box_judge(people_coords, img, dist_thres_lim=(2,)):
    num_boxes = len(people_coords)
    cv2.putText(img, f'Num:{num_boxes}', (80, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                2, (0, 255, 255), 2, cv2.LINE_AA)
    if num_boxes > dist_thres_lim[0]:
        return True, people_coords
    else:
        return False, []

def distancing(people_coords, img, dist_thres_lim=(200, 250)):
    high_risk_coordinate = []
    high_risk_report = False
    x_combs = list(itertools.combinations(people_coords, 2))
    radius = 5
    thickness = 3
    if len(dist_thres_lim) == 1:
        for x in x_combs:
            xyxy1, xyxy2 = list(map(int, x[0])), list(map(int, x[1]))
            cntr1 = ((xyxy1[2] + xyxy1[0]) // 2, (xyxy1[3] + xyxy1[1]) // 2)
            cntr2 = ((xyxy2[2] + xyxy2[0]) // 2, (xyxy2[3] + xyxy2[1]) // 2)
            dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5
            # judge by box center
            if (xyxy2[0] < cntr1[0] < xyxy2[2] and xyxy2[1] < cntr1[1] < xyxy2[3]) \
                    or (xyxy1[0] < cntr2[0] < xyxy1[2] and xyxy1[1] < cntr2[1] < xyxy1[3]):
                continue

            if 100 < dist < dist_thres_lim[0]:
                # judge by box area
                box_area1 = abs((xyxy1[0] - xyxy1[2]) * (xyxy1[1] - xyxy1[3]))
                box_area2 = abs((xyxy2[0] - xyxy2[2]) * (xyxy2[1] - xyxy2[3]))
                if (box_area1 > box_area2 and box_area1 // 2 > box_area2) or (
                        box_area1 < box_area2 and box_area1 < box_area2 // 2):
                    continue
                high_risk_coordinate.append(x)
                high_risk_report = True
                color = (0, 0, 255)
                cv2.line(img, cntr1, cntr2, color, thickness)
                cv2.circle(img, cntr1, radius, color, -1)
                cv2.circle(img, cntr2, radius, color, -1)
                # Plots one bounding box on image
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                for xy in x:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # red box

    elif len(dist_thres_lim) == 2:
        # Plot lines connecting people
        already_red = dict()  # dictionary to store if a plotted rectangle has already been labelled as high risk
        centers = []
        for i in people_coords:
            centers.append(((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2))
        for j in centers:
            already_red[j] = 0
        for x in x_combs:
            xyxy1, xyxy2 = list(map(int, x[0])), list(map(int, x[1]))
            cntr1 = ((xyxy1[2] + xyxy1[0]) // 2, (xyxy1[3] + xyxy1[1]) // 2)
            cntr2 = ((xyxy2[2] + xyxy2[0]) // 2, (xyxy2[3] + xyxy2[1]) // 2)
            dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5
            # judge by box center
            if (xyxy2[0] < cntr1[0] < xyxy2[2] and xyxy2[1] < cntr1[1] < xyxy2[3]) or (
                    xyxy1[0] < cntr2[0] < xyxy1[2] and xyxy1[1] < cntr2[1] < xyxy1[3]):
                continue

            if dist_thres_lim[0] < dist < dist_thres_lim[1]:
                color = (0, 255, 255)
                cv2.line(img, cntr1, cntr2, color, thickness)
                if already_red[cntr1] == 0:
                    cv2.circle(img, cntr1, radius, color, -1)
                if already_red[cntr2] == 0:
                    cv2.circle(img, cntr2, radius, color, -1)
                # Plots one bounding box on image img
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                for xy in x:
                    cntr = ((int(xy[2]) + int(xy[0])) // 2, (int(xy[3]) + int(xy[1])) // 2)
                    if already_red[cntr] == 0:
                        c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        # tf = max(tl - 1, 1)  # font thickness
                        # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        #             lineType=cv2.LINE_AA)

            elif 100 < dist < dist_thres_lim[0]:
                # judge by box area
                box_area1 = abs((xyxy1[0] - xyxy1[2]) * (xyxy1[1] - xyxy1[3]))
                box_area2 = abs((xyxy2[0] - xyxy2[2]) * (xyxy2[1] - xyxy2[3]))
                if (box_area1 > box_area2 and box_area1 // 2 > box_area2) or (
                        box_area1 < box_area2 and box_area1 < box_area2 // 2):
                    continue
                high_risk_coordinate.append(x)
                high_risk_report = True
                color = (0, 0, 255)
                # label = "High Risk"
                # already_red[cntr1] = 1
                # already_red[cntr2] = 1
                cv2.line(img, cntr1, cntr2, color, thickness)
                cv2.circle(img, cntr1, radius, color, -1)
                cv2.circle(img, cntr2, radius, color, -1)
                # Plots one bounding box on image
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                for xy in x:
                    c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # red box
                    # tf = max(tl - 1, 1)  # font thickness
                    # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    #             lineType=cv2.LINE_AA)
    return high_risk_report, high_risk_coordinate


def uploadftp_risk(ftp_info, image, bbox_coords, risk_coords, img_dir, txt_dir):
    # 輸出圖片&座標文字檔
    cv2.imwrite(img_dir, image)
    # # write each bbox coordinate
    for *xyxy, conf, cls in bbox_coords:
        xy_xy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
        with open(txt_dir, 'a') as file:
            file.write(('%g,' * 4 + '%g\n') % (cls, *xy_xy))
    # # write high risk bbox coordinate
    # for co in risk_coords:
    #     # print(co)
    #     xy_xy = torch.tensor(co).view(1, 8).view(-1).tolist()
    #     with open(txt_file_dir, 'a') as file:
    #         file.write(('%g ' * 9 + '\n') % (1, *xy_xy))

    ftp_up(url=ftp_info['url'], user=ftp_info['user'],
           passwd=ftp_info['passwd'], filename=[img_dir, txt_dir],
           folder=ftp_info['risk_path'], blocksize=8192)


def uploadftp_raw_data(ftp_info, images, file_name):
    file_list = []
    for i, imgs in enumerate(images):
        for j, img in enumerate(imgs):
            save_dir = f"{file_name}_C{i}{j}.jpg"
            file_list.append(save_dir)
            cv2.imwrite(save_dir, img)
    ftp_up(url=ftp_info['url'], user=ftp_info['user'],
           passwd=ftp_info['passwd'], filename=file_list,
           folder=ftp_info['raw_path'])


def uploadftp_log(ftp_info, log_file_dir):
    ftp_up(url=ftp_info['url'], user=ftp_info['user'],
           passwd=ftp_info['passwd'], filename=log_file_dir,
           folder=ftp_info['log_path'])


def ftp_up(url, user, passwd, filename, folder, blocksize=1024):
    try:
        ftp = FTP()
        ftp.set_debuglevel(0)  # 除錯級別2，顯示詳細資訊;0為關閉除錯資訊
        ftp.connect(url, 21)
        ftp.login(user, passwd)  # 登入，如果匿名登入則用空串代替即可
        # print(ftp.getwelcome()) # 顯示ftp伺服器歡迎資訊
        try:
            ftp.mkd(folder)  # 建立資料夾路徑
        except ftplib.error_perm as e:
            print(f"\033[41;37mFTP - mkd_WARNNING: {str(e)} -> {folder}\033[0m ")
        ftp.cwd(folder)  # 選擇操作目錄
        if isinstance(filename, str):
            file_handler = open(filename, 'rb')  # 以讀模式在本地開啟檔案
            # 上傳檔案  # blocksize = 1024  # 設定緩衝塊大小
            ftp.storbinary(f'STOR {os.path.basename(filename)}', file_handler, blocksize)
            file_handler.close()
        elif isinstance(filename, list):
            for file in filename:
                file_handler = open(file, 'rb')  # 以讀模式在本地開啟檔案
                # 上傳檔案  # blocksize = 1024  # 設定緩衝塊大小
                ftp.storbinary('STOR %s' % os.path.basename(file), file_handler, blocksize)
                file_handler.close()
        else:
            print(f"\033[41;37mFTP - filename_ERROR: {filename} -> {folder}\033[0m ")
            ftp.quit()
            return
        ftp.quit()
        print(f"\033[42;37mFTP - OK. File: {filename} => {folder}\033[0m ")
    except Exception as e:
        print(f"\033[41;37mFTP - ERROR:  {str(e)}\033[0m ")
        print(e)


def mask_screens(mask_info, id, img, im0s):
    for area in mask_info[id]:
        img[:,
        int(area[2] / 1080 * 384):int(area[3] / 1080 * 384),
        int(area[0] / 1920 * 640): int(area[1] / 1920 * 640)] = 0  # 384,640
        # im0s[area[2]:area[3], area[0]:area[1], :] = 0  # 1080,1920
    # if im0s:
    #     zeros1 = np.zeros((im0s.shape), dtype=np.uint8)
    #     edg_mask = cv2.rectangle(zeros1, (area[0], area[2]), (area[1], area[3]), color=(0, 0, 255), thickness=-1)
    #     try:
    #         alpha = 1
    #         beta = 0.8
    #         gamma = 0
    #         mask_img = cv2.addWeighted(im0s, alpha, edg_mask, beta, gamma)
    #         # cv2.imwrite(os.path.join(output_fold, 'mask_img.jpg'), mask_img)
    #         cv2.imshow("fig", mask_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     except:
    #         print('ERROR')


def trim_screens(trim_info, id, img):
    trim1, trim2 = trim_info[id]
    img[:, :trim1, :] = 0
    img[:, trim2:, :] = 0


def multi_screens(matrix, img, id, multi_x, multi_y):
    half_x, half_y = multi_x // 2, multi_y // 2
    im0_resize = cv2.resize(img, (half_x, half_y), interpolation=cv2.INTER_AREA)
    if id == 0:
        matrix[:half_y, :half_x, :] = im0_resize
    elif id == 1:
        matrix[:half_y, half_x:, :] = im0_resize
    elif id == 2:
        matrix[half_y:, :half_x, :] = im0_resize
    elif id == 3:
        matrix[half_y:, half_x:, :] = im0_resize
    return matrix


def save_imgs_regular(imgs, path, time):
    for i, per_im0s in enumerate(imgs):
        save_dir = Path(path, time.strftime('%Y%m%d'), f"C{i}_{time.strftime('%H%M%S')}.jpg")
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir), per_im0s)


def set_logg(log_file_dir):
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_file_dir,
                        filemode='a',
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info(f'Starting...{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
