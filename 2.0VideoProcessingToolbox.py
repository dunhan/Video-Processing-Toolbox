import datetime
import tkinter as tk
from tkinter import *
# from Utilities import *
import cv2
import numpy as np
import imagehash
from PIL import Image
import array as arr
import math
import csv
import os

# SECTION1 FUNCTIONS

def blacktowhiteratioCORE(img):
    h, w = img.shape[:2]  # Get image height & weight
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image from BGR to RGB
    r, g, b = cv2.split(RGB)
    bwp = 0  # to count number of black and white pixels
    break2 = False  # set break2 parameter to break two loop simultaneously
    for i in range(h):
        for j in range(w):
            if (r[i, j] == g[i, j] == b[i, j]):
                bwp = bwp + 1
            else:
                break2 = True
                break  # break inner for loop
        if (break2):
            break  # break outer for loop
    if (bwp == h * w):
        bw = 1
    else:
        bw = 0
    return bw
def BlacktoWhiteRatio(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    num_bw = 0  # to count number of black and white frames
    if (len(string) == 0):  # no shot changing time entered
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of Key Frame
        num_bw = num_bw + blacktowhiteratioCORE(img_old)
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5*hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame
                num_bw = num_bw + blacktowhiteratioCORE(img)
            p0 = p1  # update p1 to p0 for next loop
        BWR = num_bw / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst  = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i-1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i-1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
        for i in range(len(numloop)):
            j = 1
            while (j<=numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            # calculate if this key frame at the middle of one shot is filmed in black and white or not:
            ret, img = cap.read()  # read one frame
            if (ret == False):
                break
            ###### Core Calculation of Key Frame
            num_bw = num_bw + blacktowhiteratioCORE(img)
        BWR = num_bw / lenlst
    result1.set(BWR)
    result_tag1.set("black to white ratio (BWR):")
    result2.set("\\")
    result_tag2.set("\\")
    # print("Black to White Ratio (BWR) is ", BWR)
    # return BWR
def BlacktoWhiteRatioEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frame = 0  # to add up total number of frames
    Arr_bwr = arr.array('d', [])  # create an array to store every frame's black to white information
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        bwr = blacktowhiteratioCORE(img)  # Core calculation of black to white ratio
        Arr_bwr.append(bwr)  # store last frame's bwr
    # construct csv file to output every frame's black to white state
    headers = ['number of frame', 'current time (s)', 'black to white (1/0 represents true/false)']
    with open('BlacktoWhiteRatio.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_bwr[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's black to white information

def luminosityCORE(img):
    h = img.shape[0]  # read image height
    w = img.shape[1]  # read image weight
    img_old_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    r, g, b = cv2.split(img_old_rgb)  # Get new image RGB matrix
    lum = 0.0  # to calculate luminosity of new frame
    for i in range(h):  # Luminance=0.229*r+0.587*g+0.114*b
        for j in range(w):
            lum = lum + (0.229 * r[i, j] + 0.587 * g[i, j] + 0.114 * b[i, j])
    lum = lum / (h * w)
    return lum
def Luminosity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_lum = arr.array('d', [])  # new array to store luminosity of key frame in a shot
    if (len(string) == 0):  # no shot changing time entered
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of Key Frame
        lum = luminosityCORE(img_old)
        Arr_lum.append(lum)  # store first frame luminosity
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame
                lum = luminosityCORE(img)
                Arr_lum.append(lum)  # store key frame luminosity
            p0 = p1  # update p1 to p0 for next loop
        if (num_shots == 1):
            LUM = Arr_lum[0]
            LUV = 0.0
        else:
            LUM = np.sum(Arr_lum) / num_shots
            Var_sum = 0.0
            for i in range(len(Arr_lum)):
                Var_sum = Var_sum + (Arr_lum[i] - LUM) ** 2
            LUV = Var_sum / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(len(lst)):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret==False):
                break
            ###### Core Calculation of Key Frame
            lum = luminosityCORE(img_old)
            Arr_lum.append(lum)  # store first frame luminosity
        LUM = np.sum(Arr_lum) / lenlst
        Var_sum = 0.0
        for i in range(len(Arr_lum)):
            Var_sum = Var_sum + (Arr_lum[i] - LUM) ** 2
        LUV = Var_sum / lenlst
    result1.set(LUM)
    result_tag1.set("luminosity mean (LUM):")
    result2.set(LUV)
    result_tag2.set("luminosity variance (LUV):")
    # return LUM, LUV
def LuminosityEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    Arr_l = arr.array('d', [])  # create an array to store every frame's luminosity
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        l = luminosityCORE(img)  # Core calculation of luminosity
        Arr_l.append(l)  # store last frame's luminosity
    # construct csv file to output every frame's luminosity
    headers = ['number of frame', 'current time (s)', 'luminosity']
    with open('Luminosity.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_l[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's luminosity

def saturationCORE(img):
    h, w = img.shape[:2]  # Get image height & weight
    r, g, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))  # Get new image RGB matrix
    s1_count = 0.0
    for i in range(h):
        for j in range(w):
            maxV = max(int(r[i, j]), int(g[i, j]), int(b[i, j]))
            minV = min(int(r[i, j]), int(g[i, j]), int(b[i, j]))
            if maxV == 0:
                s = 0
            else:
                s = (maxV - minV) / maxV
            s1_count = s1_count + s  # to count Saturation of all pixels
    sm_rgb = s1_count / (h * w)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # Get new image hsv matrix
    s2_count = 0.0
    for i in range(h):
        for j in range(w):
            s2_count = s2_count + S[i, j]  # to count S component of all pixels from hsv space
    sm_hsv = s2_count / (h * w)
    return sm_rgb, sm_hsv
def Saturation(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_smrgb = arr.array('d', [])  # new array to store key frame's saturation_rgb_weighted
    Arr_smhsv = arr.array('d', [])  # new array to store key frame's saturation_hsv's S component
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        sm_rgb, sm_hsv = saturationCORE(img_old)
        Arr_smrgb.append(sm_rgb)  # store first frame saturation_rgb_weighted
        Arr_smhsv.append(sm_hsv)  # store first frame saturation_hsv_S component
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame
                sm_rgb, sm_hsv = saturationCORE(img)
                Arr_smrgb.append(sm_rgb)  # store first frame saturation_rgb_weighted
                Arr_smhsv.append(sm_hsv)  # store first frame saturation_hsv_S component
            p0 = p1  # update p1 to p0 for next loop
        SAT_rgb = np.sum(Arr_smrgb) / num_shots
        SAT_hsv = np.sum(Arr_smhsv) / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret == False):
                break
            ###### Core Calculation of Key Frame
            sm_rgb, sm_hsv = saturationCORE(img_old)
            Arr_smrgb.append(sm_rgb)  # store first frame saturation_rgb_weighted
            Arr_smhsv.append(sm_hsv)  # store first frame saturation_hsv_S component
        SAT_rgb = np.sum(Arr_smrgb) / lenlst
        SAT_hsv = np.sum(Arr_smhsv) / lenlst
    result_tag1.set("saturation (from weighted rgb mean):")
    result1.set(SAT_rgb)
    result2.set("saturation (from HSV space's S component mean):")
    result_tag2.set(SAT_hsv)
    # return SAT_rgb & SAT_hsv
def SaturationEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    Arr_smrgb = arr.array('d', [])  # create an array to store every frame's saturation from rgb space
    Arr_smhsv = arr.array('d', [])  # create an array to store every frame's saturation from hsv space
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        smrgb, smhsv = saturationCORE(img)  # Core calculation of saturation
        Arr_smrgb.append(smrgb)  # store last frame's saturation from rgb space
        Arr_smhsv.append(smhsv)  # store last frame's saturation from hsv space
    # construct csv file to output every frame's saturation
    headers = ['number of frame', 'current time (s)', 'saturation_rgb', 'saturation_hsv']
    with open('Saturation.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_smrgb[i], Arr_smhsv[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's saturation

def chromaticvarietyCORE(img, n):
    h, w = img.shape[:2]  # Get image height & weight
    r, g, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))  # Get new image RGB matrix
    Arr = np.zeros(n, dtype=int)  # to count number of a certain amount (n) of different kinds of colors
    mul1 = round(n ** (2/3))
    mul2 = round(n ** (1/3))
    deno = 256 / mul2  # denominator to section rgb spaces
    for i in range(h):
        for j in range(w):
            x = r[i, j] // deno
            y = g[i, j] // deno
            z = b[i, j] // deno
            Arr[int(mul1 * x + mul2 * y + z)] = Arr[int(mul1 * x + mul2 * y + z)] + 1  # record color occurrence
    mean_binh = np.sum(Arr) / n  # get mean bin height (color occurrence)
    # calculate variance of bins:
    cvariance_total = 0.0
    for i in range(n):
        cvariance_total = cvariance_total + i * ((Arr[i] - mean_binh) ** 2)
    VCi = cvariance_total / n
    return VCi
def ChromaticVariety(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    binnum = bin_num_entry.get()
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_vc = arr.array('d', [])  # new array to store key frame's chromatic variety
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of  First frame's Chromatic Variety
        VCi = chromaticvarietyCORE(img_old, binnum)
        Arr_vc.append(VCi)  # store first frame luminosity
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame's Chromatic Variety
                VCi = chromaticvarietyCORE(img, binnum)
                Arr_vc.append(VCi)  # store key frame chromatic variety
            p0 = p1  # update p1 to p0 for next loop
        VCavg = np.sum(Arr_vc) / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret == False):
                break
            ###### Core Calculation of Key Frame
            VCi = chromaticvarietyCORE(img_old, binnum)
            Arr_vc.append(VCi)  # store key frame chromatic variety
        VCavg = np.sum(Arr_vc) / lenlst
    result_tag1.set("chromatic variety (VCavg):")
    result1.set(VCavg)
    result2.set("\\")
    result_tag2.set("\\")
    # return VCavg
def ChromaticVarietyEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    binnum = bin_num_entry.get()
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    Arr_vci = arr.array('d', [])  # create an array to store every frame's chromatic variety
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        vci = chromaticvarietyCORE(img, binnum)  # Core calculation of chromatic variety
        Arr_vci.append(vci)  # store last frame's chromatic variety
    # construct csv file to output every frame's chromatic variety
    headers = ['number of frame', 'current time (s)', 'chromatic variety']
    with open('ChromaticVariety.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_vci[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's chromatic variety

def entropyofluminosityCORE(img, n):
    h, w = img.shape[:2]  # Get image height & weight
    img_new_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, g, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
    rgb = r  # copy a frame matrix to store value of color (rgb)
    Arr_gray = np.zeros(n, dtype=int)  # count occurrence of different gray levels
    Arr_rgb = np.zeros(n, dtype=int)  # count occurrence of different RGB levels
    mul1 = round(n ** (2 / 3))
    mul2 = round(n ** (1 / 3))
    deno = 256 / mul2  # denominator to section rgb spaces
    for i in range(h):
        for j in range(w):
            place = round(img_new_gray[i, j] / (256 / n))
            if (place == n):
                place = n - 1
            img_new_gray[i, j] = place
            Arr_gray[place] = Arr_gray[place] + 1  # count gray value occurrence
            x = r[i, j] // deno
            y = g[i, j] // deno
            z = b[i, j] // deno
            place = int(mul1 * x + mul2 * y + z)
            if (place == n):
                place = n - 1
            rgb[i, j] = place
            Arr_rgb[place] = Arr_rgb[place] + 1  # record color occurrence
    ENM_gray = 0.0  # to calculate summation of (-Pli*log2(Pli))_from gray scale space
    ENM_rgb = 0.0  # to calculate summation of (-Pli*log2(Pli))_from rgb color space
    for i in range(h):
        for j in range(w):
            P_li = (Arr_gray[img_new_gray[i, j]] / (h * w))  # probability of this pixel's gray_value in this frame
            ENM_gray = ENM_gray + (- P_li * math.log2(P_li))
            P_li = (Arr_rgb[rgb[i, j]] / (h * w))  # probability of this pixel's rgb color in this frame
            ENM_rgb  = ENM_rgb + (- P_li * math.log2(P_li))
    return ENM_gray, ENM_rgb
def EntropyofLuminosity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    Arr_ENM_GRAY = arr.array('d', [])  # new array to store key frame's entropy of luminosity from gray scale space
    Arr_ENM_RGB = arr.array('d', [])  # new array to store key frame's entropy of luminosity from rgb color space
    hashvalue = hash_value_entry.get()
    binnum = bin_num_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    string = shot_changing_entry.get()
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of Key Frame
        ENM_gray, ENM_rgb = entropyofluminosityCORE(img_old, binnum)
        Arr_ENM_GRAY.append(ENM_gray)  # store first frame's entropy of luminosity from gray scale space
        Arr_ENM_RGB.append(ENM_rgb)  # store first frame's entropy of luminosity from rgb color space
        while (1):
            ret, img = cap.read()
            if (ret == False):
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame
                ENM_gray, ENM_rgb = entropyofluminosityCORE(img, binnum)
                Arr_ENM_GRAY.append(ENM_gray)  # store key frame's entropy of luminosity from gray scale space
                Arr_ENM_RGB.append(ENM_rgb)  # store key frame's entropy of luminosity from rgb color space
            p0 = p1  # update p1 to p0 for next loop
        ENM_GRAY = np.sum(Arr_ENM_GRAY) / num_shots
        ENM_RGB  = np.sum(Arr_ENM_RGB) / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret == False):
                break
            ###### Core Calculation of Key Frame
            ENM_gray, ENM_rgb = entropyofluminosityCORE(img_old, binnum)
            Arr_ENM_GRAY.append(ENM_gray)  # store key frame's entropy of luminosity from gray scale space
            Arr_ENM_RGB.append(ENM_rgb)  # store key frame's entropy of luminosity from rgb color space
        ENM_GRAY = np.sum(Arr_ENM_GRAY) / lenlst
        ENM_RGB = np.sum(Arr_ENM_RGB) / lenlst
    result_tag1.set("entropy of luminosity (ENM) from gray scale image:")
    result1.set(ENM_GRAY)
    result_tag2.set("entropy of luminosity (ENM) from rgb color image")
    result2.set(ENM_RGB)
    # return ENM
def EntropyofLuminosityEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    binnum = bin_num_entry.get()
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    Arr_enm_gray = arr.array('d', [])  # create an array to store every frame's ENM from gray space
    Arr_enm_rgb = arr.array('d', [])  # create an array to store every frame's ENM from rgb space
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        enm_gray, enm_rgb = entropyofluminosityCORE(img, binnum)  # Core calculation of ENM
        Arr_enm_gray.append(enm_gray)  # store last frame's ENM from gray space
        Arr_enm_rgb.append(enm_rgb)  # store last frame's ENM from rgb space
    # construct csv file to output every frame's entropy of luminosity
    headers = ['number of frame', 'current time (s)', 'entropy of luminosity_gray', 'entropy of luminosity_rgb']
    with open('EntropyofLuminosity.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_enm_gray[i], Arr_enm_rgb[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's entropy of luminosity

def contrastCORE(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img.shape
    sum = 0.0
    for i in range(m):
        for j in range(n):
            sum = sum + img[i, j]
    mean_gray = sum / (m * n)  # mean gray scale
    sum_gray = 0.0
    b = 0.0
    c = 0.0
    for i in range(1, m - 1):
        # mean gray scale add up
        sum_gray = sum_gray + (img[i, 0] - mean_gray) ** 2 + (img[i, n - 1] - mean_gray) ** 2
        # four scale contrast summation-left most column:
        d = ((int(img[i, 0]) - int(img[i, 1])) ** 2 +
             (int(img[i, 0]) - int(img[i, n - 1])) ** 2 +
             (int(img[i, 0]) - int(img[i + 1, 0])) ** 2 +
             (int(img[i, 0]) - int(img[i - 1, 0])) ** 2)
        b = b + d
        # eight scale contrast summation-left most column:
        c = c + d + ((int(img[i, 0]) - int(img[i - 1, n - 1])) ** 2 +
                     (int(img[i, 0]) - int(img[i + 1, 1])) ** 2 +
                     (int(img[i, 0]) - int(img[i - 1, 1])) ** 2 +
                     (int(img[i, 0]) - int(img[i + 1, n - 1])) ** 2)
        # four scale contrast summation-right most column:
        d = ((int(img[i, n-1]) - int(img[i, 0])) ** 2 +
             (int(img[i, n-1]) - int(img[i, n - 2])) ** 2 +
             (int(img[i, n-1]) - int(img[i + 1, n-1])) ** 2 +
             (int(img[i, n-1]) - int(img[i - 1, n-1])) ** 2)
        b = b + d
        # eight scale contrast summation-right most column:
        c = c + d + ((int(img[i, n-1]) - int(img[i - 1, n - 2])) ** 2 +
                     (int(img[i, n-1]) - int(img[i + 1, 0])) ** 2 +
                     (int(img[i, n-1]) - int(img[i - 1, 0])) ** 2 +
                     (int(img[i, n-1]) - int(img[i + 1, n - 2])) ** 2)
        for j in range(1, n - 1):
            # mean gray scale add up
            sum_gray = sum_gray + (img[i, j] - mean_gray) ** 2
            # four scale contrast summation-central pixels:
            d = ((int(img[i, j]) - int(img[i, j + 1])) ** 2 +
                 (int(img[i, j]) - int(img[i, j - 1])) ** 2 +
                 (int(img[i, j]) - int(img[i + 1, j])) ** 2 +
                 (int(img[i, j]) - int(img[i - 1, j])) ** 2)
            b = b + d
            # eight scale contrast summation-central pixels:
            c = c + d + ((int(img[i, j]) - int(img[i - 1, j - 1])) ** 2 +
                         (int(img[i, j]) - int(img[i + 1, j + 1])) ** 2 +
                         (int(img[i, j]) - int(img[i - 1, j + 1])) ** 2 +
                         (int(img[i, j]) - int(img[i + 1, j - 1])) ** 2)
    for i in range(1, n-1):
        # mean gray scale add up
        sum_gray = sum_gray + (img[0, i] - mean_gray) ** 2 + (img[m - 1, i] - mean_gray) ** 2
        # four scale contrast summation-top most row:
        d = ((int(img[0, i]) - int(img[0, i - 1])) ** 2 +
             (int(img[0, i]) - int(img[0, i + 1])) ** 2 +
             (int(img[0, i]) - int(img[1, i])) ** 2 +
             (int(img[0, i]) - int(img[m-1, i])) ** 2)
        b = b + d
        # eight scale contrast summation-top most row:
        c = c + d + ((int(img[0, i]) - int(img[1, i - 1])) ** 2 +
                     (int(img[0, i]) - int(img[1, i + 1])) ** 2 +
                     (int(img[0, i]) - int(img[m - 1, i - 1])) ** 2 +
                     (int(img[0, i]) - int(img[m - 1, i + 1])) ** 2)
        # four scale contrast summation-bottom most row:
        d = ((int(img[m - 1, i]) - int(img[m - 1, i - 1])) ** 2 +
             (int(img[m - 1, i]) - int(img[m - 1, i + 1])) ** 2 +
             (int(img[m - 1, i]) - int(img[m - 2, i])) ** 2 +
             (int(img[m - 1, i]) - int(img[0, i])) ** 2)
        b = b + d
        # eight scale contrast summation-bottom most row:
        c = c + d + ((int(img[m - 1, i]) - int(img[0, i - 1])) ** 2 +
                     (int(img[m - 1, i]) - int(img[0, i + 1])) ** 2 +
                     (int(img[m - 1, i]) - int(img[m - 2, i - 1])) ** 2 +
                     (int(img[m - 1, i]) - int(img[m - 2, i + 1])) ** 2)
    # four scale contrast summation-top left most pixel:
    d = ((int(img[0, 0]) - int(img[0, 1])) ** 2 +
         (int(img[0, 0]) - int(img[0, n - 1])) ** 2 +
         (int(img[0, 0]) - int(img[1, 0])) ** 2 +
         (int(img[0, 0]) - int(img[m - 1, 0])) ** 2)
    b = b + d
    # eight scale contrast summation-top left most pixel:
    c = c + d + ((int(img[0, 0]) - int(img[1, 1])) ** 2 +
                 (int(img[0, 0]) - int(img[1, n - 1])) ** 2 +
                 (int(img[0, 0]) - int(img[m - 1, 1])) ** 2 +
                 (int(img[0, 0]) - int(img[m - 1, n - 1])) ** 2)
    # four scale contrast summation-top right most pixel:
    d = ((int(img[0, n - 1]) - int(img[0, 0])) ** 2 +
         (int(img[0, n - 1]) - int(img[0, n - 2])) ** 2 +
         (int(img[0, n - 1]) - int(img[1, n - 1])) ** 2 +
         (int(img[0, n - 1]) - int(img[m - 1, n - 1])) ** 2)
    b = b + d
    # eight scale contrast summation-top right most pixel:
    c = c + d + ((int(img[0, n - 1]) - int(img[1, 0])) ** 2 +
                 (int(img[0, n - 1]) - int(img[1, n - 2])) ** 2 +
                 (int(img[0, n - 1]) - int(img[m - 1, 0])) ** 2 +
                 (int(img[0, n - 1]) - int(img[m - 1, n - 2])) ** 2)
    # four scale contrast summation-bottom right most pixel:
    d = ((int(img[m - 1, n - 1]) - int(img[0, n - 1])) ** 2 +
         (int(img[m - 1, n - 1]) - int(img[m - 2, n - 1])) ** 2 +
         (int(img[m - 1, n - 1]) - int(img[m - 1, 0])) ** 2 +
         (int(img[m - 1, n - 1]) - int(img[m - 1, n - 2])) ** 2)
    b = b + d
    # eight scale contrast summation-bottom right most pixel:
    c = c + d + ((int(img[m - 1, n - 1]) - int(img[0, 0])) ** 2 +
                 (int(img[m - 1, n - 1]) - int(img[0, n - 2])) ** 2 +
                 (int(img[m - 1, n - 1]) - int(img[m - 2, 0])) ** 2 +
                 (int(img[m - 1, n - 1]) - int(img[m - 2, n - 2])) ** 2)
    # four scale contrast summation-bottom left most pixel:
    d = ((int(img[m - 1, 0]) - int(img[0, 0])) ** 2 +
         (int(img[m - 1, 0]) - int(img[m - 2, 0])) ** 2 +
         (int(img[m - 1, 0]) - int(img[m - 1, 1])) ** 2 +
         (int(img[m - 1, 0]) - int(img[m - 1, n - 1])) ** 2)
    b = b + d
    # eight scale contrast summation-bottom left most pixel:
    c = c + d + ((int(img[m - 1, 0]) - int(img[0, 1])) ** 2 +
                 (int(img[m - 1, 0]) - int(img[0, n - 1])) ** 2 +
                 (int(img[m - 1, 0]) - int(img[m - 2, 1])) ** 2 +
                 (int(img[m - 1, 0]) - int(img[m - 2, n - 1])) ** 2)
    # four corners mean gray contrast add up:
    sum_gray = sum_gray + (img[0, 0] - mean_gray) ** 2 + (img[m - 1, n - 1] - mean_gray) ** 2 + \
               (img[0, n - 1] - mean_gray) ** 2 + (img[m - 1, 0] - mean_gray) ** 2
    fc = b / (4 * m * n)  # four scale contrast
    ec = c / (8 * m * n)  # eight scale contrast
    gc = math.sqrt(sum_gray / (m * n))  # mean gray contrast / Peli contrast
    return fc, ec, gc
def Contrast(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    Arr_fc = arr.array('d', [])  # new array to store key frame's four scale contrast
    Arr_ec = arr.array('d', [])  # new array to store key frame's eight scale contrast
    Arr_gc = arr.array('d', [])  # new array to store key frame's Peli (mean gray) contrast
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of first Frame
        fc, ec, gc = contrastCORE(img_old)
        Arr_fc.append(fc)  # store first frame four scale contrast
        Arr_ec.append(ec)  # store first frame eight scale contrast
        Arr_gc.append(gc)  # store first frame Peli contrast
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame
                fc, ec, gc = contrastCORE(img)
                Arr_fc.append(fc)  # store first frame four scale contrast
                Arr_ec.append(ec)  # store first frame eight scale contrast
                Arr_gc.append(gc)  # store first frame Peli contrast
            p0 = p1  # update p1 to p0 for next loop
        fc_mean = np.sum(Arr_fc) / num_shots
        ec_mean = np.sum(Arr_ec) / num_shots
        gc_mean = np.sum(Arr_gc) / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret == False):
                break
            ###### Core Calculation of Key Frame
            fc, ec, gc = contrastCORE(img_old)
            Arr_fc.append(fc)  # store first frame four scale contrast
            Arr_ec.append(ec)  # store first frame eight scale contrast
            Arr_gc.append(gc)  # store first frame Peli contrast
        fc_mean = np.sum(Arr_fc) / lenlst
        ec_mean = np.sum(Arr_ec) / lenlst
        gc_mean = np.sum(Arr_gc) / lenlst
    outa = tuple([fc_mean, ec_mean])
    result1.set(outa)
    result_tag1.set("four scale contrast mean, eight scale contrast mean:")
    result2.set(gc_mean)
    result_tag2.set("Peli (mean gray) contrast:")
    # return fc_mean, ec_mean, gc_mean
def ContrastEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    Arr_fc = arr.array('d', [])  # create an array to store every frame's four-scale contrast
    Arr_ec = arr.array('d', [])  # create an array to store every frame's eight-scale contrast
    Arr_gc = arr.array('d', [])  # create an array to store every frame's mean gray contrast
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        fc, ec, gc = contrastCORE(img)  # Core calculation of contrast
        Arr_fc.append(fc)  # store last frame's four-scale contrast
        Arr_ec.append(ec)  # store last frame's eight-scale contrast
        Arr_gc.append(gc)  # store last frame's mean gray contrast
    # construct csv file to output every frame's contrast
    headers = ['number of frame', 'current time (s)', 'four-scale contrast', 'eight-scale contrast',
               'mean gray contrast / Peli contrast']
    with open('Contrast.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_fc[i], Arr_ec[i], Arr_gc[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file that contain every frame's contrast

def LowLevelFeaturesALL(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    Arr_fc = arr.array('d', [])  # new array to store key frame's four scale contrast
    Arr_ec = arr.array('d', [])  # new array to store key frame's eight scale contrast
    Arr_gc = arr.array('d', [])  # new array to store key frame's Peli (mean gray) contrast
    Arr_ENM_GRAY = arr.array('d', [])  # new array to store key frame's entropy of luminosity from gray scale space
    Arr_ENM_RGB = arr.array('d', [])  # new array to store key frame's entropy of luminosity from rgb color space
    Arr_vc = arr.array('d', [])  # new array to store key frame's chromatic variety
    Arr_smrgb = arr.array('d', [])  # new array to store key frame's saturation_rgb_weighted
    Arr_smhsv = arr.array('d', [])  # new array to store key frame's saturation_hsv's S component
    Arr_lum = arr.array('d', [])  # new array to store key frame's luminosity
    num_bw = 0.0  # to count number of shot filmed in black and white
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    binnum = bin_num_entry.get()
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, img_old = cap.read()
        PIL0 = Image.fromarray(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))  # convert old frame to RGB
        p0 = imagehash.average_hash(PIL0, hash_size=9)  # old image hash value
        num_shots = 1  # to count number of shots
        ###### Core Calculation of Key Frame's Contrast
        fc, ec, gc = contrastCORE(img_old)
        Arr_fc.append(fc)  # store first frame four scale contrast
        Arr_ec.append(ec)  # store first frame eight scale contrast
        Arr_gc.append(gc)  # store first frame Peli contrast
        ###### Core Calculation of Key Frame
        ENM_gray, ENM_rgb = entropyofluminosityCORE(img_old, binnum)
        Arr_ENM_GRAY.append(ENM_gray)  # store first frame's entropy of luminosity from gray scale space
        Arr_ENM_RGB.append(ENM_rgb)  # store first frame's entropy of luminosity from rgb color space
        ###### Core Calculation of Key Frame's Chromatic Variety
        VCi = chromaticvarietyCORE(img_old, binnum)
        Arr_vc.append(VCi)  # store first frame chromatic variety
        ###### Core Calculation of Key Frame's Saturation
        sm_rgb, sm_hsv = saturationCORE(img_old)
        Arr_smrgb.append(sm_rgb)  # store first frame saturation_rgb_weighted
        Arr_smhsv.append(sm_hsv)  # store first frame saturation_hsv_S component
        ###### Core Calculation of Key Frame's luminosity
        lum = luminosityCORE(img_old)
        Arr_lum.append(lum)  # store first frame luminosity
        ###### Core Calculation of Key Frame's Black to White Ratio
        num_bw = num_bw + blacktowhiteratioCORE(img_old)
        while (1):
            ret, img = cap.read()
            if ret == False:
                break
            PIL1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert new frame to RGB
            p1 = imagehash.average_hash(PIL1, hash_size=9)  # new image hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # detecting shot changing
                ###### Core Calculation of Key Frame's Contrast
                fc, ec, gc = contrastCORE(img)
                Arr_fc.append(fc)  # store key frame four scale contrast
                Arr_ec.append(ec)  # store key frame eight scale contrast
                Arr_gc.append(gc)  # store key frame Peli contrast
                ###### Core Calculation of Key Frame
                ENM_gray, ENM_rgb = entropyofluminosityCORE(img, binnum)
                Arr_ENM_GRAY.append(ENM_gray)  # store key frame's entropy of luminosity from gray scale space
                Arr_ENM_RGB.append(ENM_rgb)  # store key frame's entropy of luminosity from rgb color space
                ###### Core Calculation of Key Frame's Chromatic Variety
                VCi = chromaticvarietyCORE(img, binnum)
                Arr_vc.append(VCi)  # store key frame chromatic variety
                ###### Core Calculation of Key Frame's Saturation
                sm_rgb, sm_hsv = saturationCORE(img)
                Arr_smrgb.append(sm_rgb)  # store key frame saturation_rgb_weighted
                Arr_smhsv.append(sm_hsv)  # store key frame saturation_hsv_S component
                ###### Core Calculation of Key Frame's luminosity
                lum = luminosityCORE(img)
                Arr_lum.append(lum)  # store key frame luminosity
                ###### Core Calculation of Key Frame's Black to White Ratio
                num_bw = num_bw + blacktowhiteratioCORE(img)
            p0 = p1  # update p1 to p0 for next loop
        fc_mean = np.sum(Arr_fc) / num_shots
        ec_mean = np.sum(Arr_ec) / num_shots
        gc_mean = np.sum(Arr_gc) / num_shots
        ENM_GRAY = np.sum(Arr_ENM_GRAY) / num_shots
        ENM_RGB = np.sum(Arr_ENM_RGB) / num_shots
        VCavg = np.sum(Arr_vc) / num_shots
        SAT_rgb = np.sum(Arr_smrgb) / num_shots
        SAT_hsv = np.sum(Arr_smhsv) / num_shots
        if (num_shots == 1):
            LUM = Arr_lum[0]
            LUV = 0.0
        else:
            LUM = np.sum(Arr_lum) / num_shots
            Var_sum = 0.0
            for i in range(len(Arr_lum)):
                Var_sum = Var_sum + (Arr_lum[i] - LUM) ** 2
            LUV = Var_sum / num_shots
        BWR = num_bw / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        numloop = arr.array('d', [])
        for i in range(lenlst):
            if (i != 0):
                numloop.append((lst[i] - lst[i - 1]) * fps / 2 + LastFramsToRead)
                LastFramsToRead = (lst[i] - lst[i - 1]) * fps / 2
            else:
                LastFramsToRead = lst[i] * fps / 2
                numloop.append(LastFramsToRead)
            # print(numloop[i], "numloop[", i, "]")
        for i in range(len(numloop)):
            j = 1
            while (j <= numloop[i]):
                ret, img = cap.read()  # read one frame
                if (ret == False):
                    break
                j = j + 1
            ret, img_old = cap.read()
            if (ret == False):
                break
            ###### Core Calculation of Key Frame's Contrast
            fc, ec, gc = contrastCORE(img_old)
            Arr_fc.append(fc)  # store first frame four scale contrast
            Arr_ec.append(ec)  # store first frame eight scale contrast
            Arr_gc.append(gc)  # store first frame Peli contrast
            ###### Core Calculation of Key Frame
            ENM_gray, ENM_rgb = entropyofluminosityCORE(img_old, binnum)
            Arr_ENM_GRAY.append(ENM_gray)  # store first frame's entropy of luminosity from gray scale space
            Arr_ENM_RGB.append(ENM_rgb)  # store first frame's entropy of luminosity from rgb color space
            ###### Core Calculation of Key Frame's Chromatic Variety
            VCi = chromaticvarietyCORE(img_old, binnum)
            Arr_vc.append(VCi)  # store key frame chromatic variety
            ###### Core Calculation of Key Frame's Saturation
            sm_rgb, sm_hsv = saturationCORE(img_old)
            Arr_smrgb.append(sm_rgb)  # store key frame saturation_rgb_weighted
            Arr_smhsv.append(sm_hsv)  # store key frame saturation_hsv_S component
            ###### Core Calculation of Key Frame's luminosity
            lum = luminosityCORE(img_old)
            Arr_lum.append(lum)  # store key frame luminosity
            ###### Core Calculation of Key Frame's Black to White Ratio
            num_bw = num_bw + blacktowhiteratioCORE(img_old)
        fc_mean = np.sum(Arr_fc) / lenlst
        ec_mean = np.sum(Arr_ec) / lenlst
        gc_mean = np.sum(Arr_gc) / lenlst
        ENM_GRAY = np.sum(Arr_ENM_GRAY) / lenlst
        ENM_RGB = np.sum(Arr_ENM_RGB) / lenlst
        VCavg = np.sum(Arr_vc) / lenlst
        SAT_rgb = np.sum(Arr_smrgb) / lenlst
        SAT_hsv = np.sum(Arr_smhsv) / lenlst
        LUM = np.sum(Arr_lum) / lenlst
        Var_sum = 0.0
        for i in range(len(Arr_lum)):
            Var_sum = Var_sum + (Arr_lum[i] - LUM) ** 2
        LUV = Var_sum / lenlst
        BWR = num_bw / lenlst
    outa = tuple([BWR, LUM, LUV, SAT_rgb, SAT_hsv])
    outb = tuple([VCavg, ENM_GRAY, ENM_RGB, fc_mean, ec_mean, gc_mean])
    result1.set(outa)
    result_tag1.set("Black to White Ratio; Luminosity-average; Luminosity-variance; "
                    "Saturation-weighted rgb; Saturation-S in HSV space")
    result2.set(outb)
    result_tag2.set("Chromatic Variety; Entropy of Luminosity (gray image); Entropy of Luminosity (rgb image); "
                    "Four Scale Contrast; Eight Scale Contrast; Peli (mean gray) Contrast:")
    # return ALL LOW LEVEL FEATURES
def LowLevelFeaturesALLEF(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return  # above: use to open video
    binnum = bin_num_entry.get()
    if (len(binnum) == 0):
        binnum = 256
    else:
        binnum = int(binnum)
    num_frame = 0  # to add up total number of frames
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    # construct arrays for csv writer
    Arr_bwr = arr.array('d', [])  # create an array to store every frame's black to white information
    Arr_l = arr.array('d', [])  # create an array to store every frame's luminosity
    Arr_smrgb = arr.array('d', [])  # create an array to store every frame's saturation from rgb space
    Arr_smhsv = arr.array('d', [])  # create an array to store every frame's saturation from hsv space
    Arr_vci = arr.array('d', [])  # create an array to store every frame's chromatic variety
    Arr_enm_gray = arr.array('d', [])  # create an array to store every frame's ENM from gray space
    Arr_enm_rgb = arr.array('d', [])  # create an array to store every frame's ENM from rgb space
    Arr_fc = arr.array('d', [])  # create an array to store every frame's four-scale contrast
    Arr_ec = arr.array('d', [])  # create an array to store every frame's eight-scale contrast
    Arr_gc = arr.array('d', [])  # create an array to store every frame's mean gray contrast
    while (1):
        ret, img = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        ###### Core Calculation
        bwr = blacktowhiteratioCORE(img)  # Core calculation of black to white ratio
        Arr_bwr.append(bwr)  # store last frame's bwr
        l = luminosityCORE(img)  # Core calculation of luminosity
        Arr_l.append(l)  # store last frame's luminosity
        smrgb, smhsv = saturationCORE(img)  # Core calculation of saturation
        Arr_smrgb.append(smrgb)  # store last frame's saturation from rgb space
        Arr_smhsv.append(smhsv)  # store last frame's saturation from hsv space
        vci = chromaticvarietyCORE(img, binnum)  # Core calculation of chromatic variety
        Arr_vci.append(vci)  # store last frame's chromatic variety
        enm_gray, enm_rgb = entropyofluminosityCORE(img, binnum)  # Core calculation of ENM
        Arr_enm_gray.append(enm_gray)  # store last frame's ENM from gray space
        Arr_enm_rgb.append(enm_rgb)  # store last frame's ENM from rgb space
        fc, ec, gc = contrastCORE(img)  # Core calculation of contrast
        Arr_fc.append(fc)  # store last frame's four-scale contrast
        Arr_ec.append(ec)  # store last frame's eight-scale contrast
        Arr_gc.append(gc)  # store last frame's mean gray contrast
    # construct a csv file to output every frame's all image features
    headers = ['number of frame', 'current time (s)', 'black to white (1/0 represents true/false)', 'luminosity',
               'saturation_rgb', 'saturation_hsv', 'chromatic variety', 'entropy of luminosity_gray',
               'entropy of luminosity_rgb', 'four-scale contrast', 'eight-scale contrast',
               'mean gray contrast / Peli contrast']
    with open('ImageFeaturesALL.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(num_frame):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_bwr[i], Arr_l[i], Arr_smrgb[i], Arr_smhsv[i], Arr_vci[i],
                            Arr_enm_gray[i], Arr_enm_rgb[i], Arr_fc[i], Arr_ec[i], Arr_gc[i]])
    result1.set("csv file successfully constructed,")
    result_tag1.set("Please check at current folder!")
    result2.set("\\")
    result_tag2.set("\\")
    # construct a csv file to output every frame's all image features


# SECTION2 FUNCTIONS

def AverageShotLength(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        ret, old_frame = cap.read()  # get first frame
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value calculation
        fps = cap.get(cv2.CAP_PROP_FPS)  # get video fps
        num_shots = 1  # define original number of shots
        num_frame = 1  # define original number of frames
        ArrShotTime = arr.array('d', [])  # Create an array for storing each shot duration
        LastTime = 0.0  # Set original cutoff time
        Arr_SCsites = arr.array('d', [])  # Create an array for storing shot changing sites

        while (1):  # loop doesn't end until break
            ret, frame = cap.read()  # read next frame
            if ret == False:  # no frame read
                break
            new_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
            num_frame = num_frame + 1  # one more frame read
            p1 = imagehash.average_hash(new_PIL, hash_size=9)
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                Arr_SCsites.append(num_frame)  # store number of frame at this shot changing site
                num_shots = num_shots + 1  # one more shot
                time = num_frame / fps  # time when switching shots
                # print("number of frame is ", num_frame, "time now is ", time)
                duration = time - LastTime  # duration of last shot
                ArrShotTime.append(duration)  # store duration of current shot
                LastTime = time  # set last time for next shot cutoff time
            p0 = p1  # update the hash value just calculated to old hash value
        Video_length = (num_frame / fps)  # get video length_in seconds
        Avg_shot_length = Video_length / num_shots  # calculate average shot length
        # construct csv file to output all shot changing site
        headers = ['number of frame', 'current time (s)']
        with open('Shot_Changing_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
                  'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            for i in range(len(Arr_SCsites)):
                sc_f = Arr_SCsites[i]
                f_csv.writerow([sc_f,  sc_f / fps])
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        Arr_shotlength = arr.array('d', [])  # create new array to store each shot length
        for i in range(len(lst)):
            if (i!=0):
                Arr_shotlength.append(lst[i] - lst[i-1])
            else:
                Arr_shotlength.append(lst[i])
        Avg_shot_length = np.sum(Arr_shotlength) / len(Arr_shotlength)
    result_tag1.set("Average shot length:")
    result1.set(Avg_shot_length)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return Avg_shot_length
def VarianceOfShotLength(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        ret, old_frame = cap.read()  # get first frame
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old frame calculation
        fps = cap.get(cv2.CAP_PROP_FPS)  # get video fps
        num_shots = 1  # define original number of shots
        num_frame = 1  # define original number of frames
        ArrShotTime = arr.array('d', [])  # Create an array for storing each shot duration
        LastTime = 0.0  # Set original cutoff time
        Arr_SCsites = arr.array('d', [])  # Create an array for storing shot changing sites

        while (1):  # loop doesn't end until break
            ret, frame = cap.read()  # read next frame
            if ret == False:  # no frame read
                break
            new_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
            num_frame = num_frame + 1  # one more frame read
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new frame calculation
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                Arr_SCsites.append(num_frame)  # store number of frame at this shot changing site
                num_shots = num_shots + 1  # one more shot
                time = num_frame / fps  # time when switching shots
                duration = time - LastTime  # duration of last shot
                ArrShotTime.append(duration)  # store duration of current shot
                LastTime = time  # set last time for next shot cutoff time
            p0 = p1  # update the hash value just calculated to old hash value
        Video_length = (num_frame / fps)  # get video length_in seconds
        duration = Video_length - LastTime  # calculate the last shot length
        ArrShotTime.append(duration)  # store the last duration
        Avg_shot_length = Video_length / num_shots  # calculate average shot length
        Variance_sum = 0.0  # define sum of all squares
        arrLength = len(ArrShotTime)  # Get array length
        for i in range(arrLength):
            Variance_sum = Variance_sum + (((ArrShotTime[i] - Avg_shot_length) ** 2))
        variance = Variance_sum / num_shots
        # construct csv file to output all shot changing site
        headers = ['number of frame', 'current time (s)']
        with open('Shot_Changing_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
                  'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            for i in range(len(Arr_SCsites)):
                sc_f = Arr_SCsites[i]
                f_csv.writerow([sc_f,  sc_f / fps])
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        Arr_shotlength = arr.array('d', [])  # create new array to store each shot length
        for i in range(len(lst)):
            if (i != 0):
                Arr_shotlength.append(lst[i] - lst[i - 1])
            else:
                Arr_shotlength.append(lst[i])
        Avg_shot_length = np.sum(Arr_shotlength) / len(Arr_shotlength)
        Variance_sum = 0.0
        for i in range(len(Arr_shotlength)):
            Variance_sum = Variance_sum + (Arr_shotlength[i] - Avg_shot_length) ** 2
        variance = Variance_sum / len(Arr_shotlength)
    result_tag1.set("variance of shot length:")
    result1.set(variance)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return variance
def MedianShotLength(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        ret, old_frame = cap.read()  # get first frame
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # old frame hash value
        fps = cap.get(cv2.CAP_PROP_FPS)  # get video fps
        num_shots = 1  # define original number of shots
        num_frame = 1  # define original number of frames
        ArrShotTime = arr.array('d', [])  # Create an array for storing each shot duration
        LastTime = 0.0  # Set original cutoff time
        Arr_SCsites = arr.array('d', [])  # Create an array for storing shot changing sites

        while (1):  # loop doesn't end until break
            ret, frame = cap.read()  # read next frame
            if ret == False:  # no frame read
                break
            new_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
            num_frame = num_frame + 1  # one more frame read
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # new frame hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                Arr_SCsites.append(num_frame)  # store number of frame at this shot changing site
                num_shots = num_shots + 1  # one more shot
                time = num_frame / fps  # time when switching shots
                duration = time - LastTime  # duration of last shot
                ArrShotTime.append(duration)  # store duration of current shot
                LastTime = time  # set last time for next shot cutoff time
            p0 = p1  # update the hash value just calculated to old hash value
        Video_length = (num_frame / fps)  # get video length_in seconds
        duration = Video_length - LastTime  # calculate the last shot length
        ArrShotTime.append(duration)  # store the last duration
        SmalltoLarge_Arr = sorted(ArrShotTime)
        NewArr_length = len(SmalltoLarge_Arr)
        # Calculate Median Shot Length depending on the odd/even value of its length
        if (NewArr_length % 2) == 0:  # if even, use the mean value of the central two
            a = int((NewArr_length / 2) - 1)
            b = int(NewArr_length / 2)
            Median_Shot_Length = (SmalltoLarge_Arr[a] + SmalltoLarge_Arr[b]) / 2
        else:  # else odd, get the median value directly
            Median_Shot_Length = SmalltoLarge_Arr[NewArr_length // 2]
        # construct csv file to output all shot changing site
        headers = ['number of frame', 'current time (s)']
        with open('Shot_Changing_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
                  'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            for i in range(len(Arr_SCsites)):
                sc_f = Arr_SCsites[i]
                f_csv.writerow([sc_f, sc_f / fps])
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        Arr_shotlength = arr.array('d', [])  # create new array to store each shot length
        for i in range(len(lst)):
            if (i != 0):
                Arr_shotlength.append(lst[i] - lst[i - 1])
            else:
                Arr_shotlength.append(lst[i])
        Arr_shotlength_sorted = sorted(Arr_shotlength)
        NewArr_length = len(Arr_shotlength_sorted)
        # Calculate Median Shot Length depending on the odd/even value of its length
        if (NewArr_length % 2) == 0:  # if even, use the mean value of the central two
            a = int((NewArr_length / 2) - 1)
            b = int(NewArr_length / 2)
            Median_Shot_Length = (Arr_shotlength_sorted[a] + Arr_shotlength_sorted[b]) / 2
        else:  # else odd, get the median value directly
            Median_Shot_Length = Arr_shotlength_sorted[NewArr_length // 2]
    result_tag1.set("Median_Shot_Length:")
    result1.set(Median_Shot_Length)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return Median_Shot_Length

def shotlengthALL(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        ret, old_frame = cap.read()  # get first frame
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # old frame hash value
        fps = cap.get(cv2.CAP_PROP_FPS)  # get video fps
        num_shots = 1  # define original number of shots
        num_frame = 1  # define original number of frames
        ArrShotTime = arr.array('d', [])  # Create an array for storing each shot duration
        LastTime = 0.0  # Set original cutoff time
        Arr_SCsites = arr.array('d', [])  # Create an array for storing shot changing sites

        while (1):  # loop doesn't end until break
            ret, frame = cap.read()  # read next frame
            if ret == False:  # no frame read
                break
            new_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # convert to PIL Image
            num_frame = num_frame + 1  # one more frame read
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # new frame hash value
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                Arr_SCsites.append(num_frame)  # store number of frame at this shot changing site
                num_shots = num_shots + 1  # one more shot
                time = num_frame / fps  # time when switching shots
                duration = time - LastTime  # duration of last shot
                ArrShotTime.append(duration)  # store duration of current shot
                LastTime = time  # set last time for next shot cutoff time
            p0 = p1  # update the hash value just calculated to old hash value
        Video_length = (num_frame / fps)  # get video length_in seconds
        duration = Video_length - LastTime  # calculate the last shot length
        ArrShotTime.append(duration)  # store the last duration
        Avg_shot_length = Video_length / num_shots  # calculate average shot length
        Variance_sum = 0.0  # define sum of all squares
        arrLength = len(ArrShotTime)  # Get array length
        for i in range(arrLength):
            Variance_sum = Variance_sum + (((ArrShotTime[i] - Avg_shot_length) ** 2))
        variance = Variance_sum / num_shots  # calculate variance of shot length
        SmalltoLarge_Arr = sorted(ArrShotTime)
        NewArr_length = len(SmalltoLarge_Arr)
        # Calculate Median Shot Length depending on the odd/even value of its length
        if (NewArr_length % 2) == 0:  # if even, use the mean value of the central two
            a = int((NewArr_length / 2) - 1)
            b = int(NewArr_length / 2)
            Median_Shot_Length = (SmalltoLarge_Arr[a] + SmalltoLarge_Arr[b]) / 2
        else:  # else odd, get the median value directly
            Median_Shot_Length = SmalltoLarge_Arr[NewArr_length // 2]
        # construct csv file to output all shot changing site
        headers = ['number of frame', 'current time (s)']
        with open('Shot_Changing_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
                  'w', newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            for i in range(len(Arr_SCsites)):
                sc_f = Arr_SCsites[i]
                f_csv.writerow([sc_f, sc_f / fps])
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        Arr_shotlength = arr.array('d', [])  # create new array to store each shot length
        for i in range(len(lst)):
            if (i != 0):
                Arr_shotlength.append(lst[i] - lst[i - 1])
            else:
                Arr_shotlength.append(lst[i])
        Avg_shot_length = np.sum(Arr_shotlength) / len(Arr_shotlength)
        Variance_sum = 0.0
        for i in range(len(Arr_shotlength)):
            Variance_sum = Variance_sum + (Arr_shotlength[i] - Avg_shot_length) ** 2
        variance = Variance_sum / len(Arr_shotlength)  # calculate variance of shot length
        Arr_shotlength_sorted = sorted(Arr_shotlength)
        NewArr_length = len(Arr_shotlength_sorted)
        # Calculate Median Shot Length depending on the odd/even value of its length
        if (NewArr_length % 2) == 0:  # if even, use the mean value of the central two
            a = int((NewArr_length / 2) - 1)
            b = int(NewArr_length / 2)
            Median_Shot_Length = (Arr_shotlength_sorted[a] + Arr_shotlength_sorted[b]) / 2
        else:  # else odd, get the median value directly
            Median_Shot_Length = Arr_shotlength_sorted[NewArr_length // 2]
    outa = tuple([Avg_shot_length, variance, Median_Shot_Length])
    result_tag1.set("Average Shot Length; Variance of Shot Length, Median Shot Length:")
    result1.set(outa)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")

def runtime(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    if cap.isOpened():  # when cap.isOpened()successful return True, else False
        fps = cap.get(5)  # get fps
        num_frame = cap.get(7)  # get total number of frames
    duration = num_frame / fps  # in seconds
    result_tag1.set("video's length:")
    result1.set(duration)
    result2.set("\\")
    result_tag2.set("\\")
    # return duration

def FadeRate(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 19
    else:
        hashvalue = float(hashvalue)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frame = 0
    num_FO = 0
    num_FI = 0
    last_FO = 0
    last_FI = 0
    num_shots = 1
    last_time = 0
    ret, img = cap.read()
    num_frame = num_frame + 1
    PIL0 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert to PIL Image
    s0 = imagehash.average_hash(PIL0)
    Arr_FIsites = arr.array('d', [])  # Create an array for storing fade in sites
    Arr_FOsites = arr.array('d', [])  # Create an array for storing fade out sites
    while (1):
        ret, img1_old = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        ret, img2_old = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        ret, img3_old = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        ret, img4_old = cap.read()
        if ret == False:
            break
        num_frame = num_frame + 1
        img1 = cv2.cvtColor(img1_old, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2_old, cv2.COLOR_BGR2GRAY)
        img3 = cv2.cvtColor(img3_old, cv2.COLOR_BGR2GRAY)
        img4 = cv2.cvtColor(img4_old, cv2.COLOR_BGR2GRAY)
        PIL1 = Image.fromarray(cv2.cvtColor(img1_old, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        PIL2 = Image.fromarray(cv2.cvtColor(img2_old, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        PIL3 = Image.fromarray(cv2.cvtColor(img3_old, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        PIL4 = Image.fromarray(cv2.cvtColor(img4_old, cv2.COLOR_BGR2RGB))  # convert to PIL Image
        s1 = imagehash.average_hash(PIL1)
        s2 = imagehash.average_hash(PIL2)
        s3 = imagehash.average_hash(PIL3)
        s4 = imagehash.average_hash(PIL4)
        if (abs(s0 - s1) >= hashvalue or abs(s1 - s2) >= hashvalue or abs(s2 - s3) >= hashvalue or
                abs(s3 - s4) >= hashvalue):
            time = num_frame / fps  # time when switching shots
            duration = time - last_time
            if duration >= 0.4:
                num_shots = num_shots + 1  # one more shot
                print("detecting shot changing at frame ", num_frame, "time now is ", time)
                last_time = time
        s0 = s4
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)  # ShiTomasi Set parameters
        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)  # Get corner points
        p1 = cv2.goodFeaturesToTrack(img2, mask=None, **feature_params)  # Get corner points
        p2 = cv2.goodFeaturesToTrack(img3, mask=None, **feature_params)  # Get corner points
        p3 = cv2.goodFeaturesToTrack(img4, mask=None, **feature_params)  # Get corner points
        a = img1.mean()
        b = img2.mean()
        c = img3.mean()
        d = img4.mean()
        if d < c < b < a:
            if np.array_equal(p0, p1) and np.array_equal(p1, p2) and np.array_equal(p2, p3):
                if num_frame != (last_FO + 4) and num_frame != (last_FO + 8):
                    if abs(a + b - c - d) >= 1:
                        time = num_frame / fps
                        print("detecting fade out at frame", num_frame, "at time ", time, "(seconds)")
                        Arr_FOsites.append(num_frame)  # store the number of frame at this fade out site
                        num_FO = num_FO + 1
                        last_FO = num_frame
                        print(a, b, c, d)
        if a < b < c < d:
            if np.array_equal(p0, p1) and np.array_equal(p1, p2) and np.array_equal(p2, p3):
                if num_frame != (last_FI + 4) and num_frame != (last_FI + 8):
                    if abs(a + b - c - d) >= 1:
                        time = num_frame / fps
                        print("detecting fade in at frame", num_frame, "at time ", time, "(seconds)")
                        Arr_FIsites.append(num_frame)  # store the number of frame at this fade in site
                        num_FI = num_FI + 1
                        last_FI = num_frame
                        print(a, b, c, d)
    if num_shots == 1:
        FAR = 0
    else:
        FAR = (num_FI + num_FO) / (num_shots - 1)
    # construct csv file to output all fade event sites
    headers = ['number of frame', 'current time (s)', 'fade event type']
    with open('Fade_Event_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
              'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_FIsites)):
            f_f = Arr_FIsites[i]
            f_csv.writerow([f_f, f_f / fps, 'fade in'])
        for i in range(len(Arr_FOsites)):
            f_f = Arr_FOsites[i]
            f_csv.writerow([f_f, f_f / fps, 'fade out'])
    result_tag1.set("Fade Rate:")
    result1.set(FAR)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return FAR

def DissolveRate(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    video = cv2.VideoCapture(route)
    if video.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 19
    else:
        hashvalue = float(hashvalue)
    num_frame = 0  # set parameter to count total frame
    num_Dissolve = 0  # set parameter to count total number of dissolve events detected
    fps = float(video.get(cv2.CAP_PROP_FPS))  # get video fps
    num_shots = 1  # set parameter to count total number of shots detected
    last_time = 0  # record last time when detecting shots changing
    Arr_Dsites = arr.array('d', [])  # Create an array for storing dissolve sites
    while (1):  # loop doesn't end until break
        ret, frame1_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read first frame
        ret, frame2_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read second frame
        ret, frame3_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read third frame
        ret, frame4_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read fourth frame
        ret, frame5_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read fifth frame
        ret, frame6_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read fifth frame
        ret, frame7_old = video.read()
        if ret == False:
            break
        num_frame = num_frame + 1  # read fifth frame

        # calculate hash value of each frame
        frame1 = Image.fromarray(cv2.cvtColor(frame1_old, cv2.COLOR_BGR2RGB))
        frame2 = Image.fromarray(cv2.cvtColor(frame2_old, cv2.COLOR_BGR2RGB))
        frame3 = Image.fromarray(cv2.cvtColor(frame3_old, cv2.COLOR_BGR2RGB))
        frame4 = Image.fromarray(cv2.cvtColor(frame4_old, cv2.COLOR_BGR2RGB))
        frame5 = Image.fromarray(cv2.cvtColor(frame5_old, cv2.COLOR_BGR2RGB))
        frame6 = Image.fromarray(cv2.cvtColor(frame6_old, cv2.COLOR_BGR2RGB))
        frame7 = Image.fromarray(cv2.cvtColor(frame7_old, cv2.COLOR_BGR2RGB))
        s1 = imagehash.average_hash(frame1)
        s2 = imagehash.average_hash(frame2)
        s3 = imagehash.average_hash(frame3)
        s4 = imagehash.average_hash(frame4)
        s5 = imagehash.average_hash(frame5)
        s6 = imagehash.average_hash(frame6)
        s7 = imagehash.average_hash(frame7)
        if (num_frame == 7):  # set initial value when first go through this while loop
            s0 = s1

        # detecting shots changing by hash value, using threshold value of 19
        if (abs(s0 - s1) >= hashvalue or abs(s1 - s2) >= hashvalue or abs(s2 - s3) >= hashvalue or
                abs(s3 - s4) >= hashvalue or abs(s4 - s5) >= hashvalue or
                abs(s5 - s6) >= hashvalue or abs(s6 - s7) >= hashvalue):
            time = num_frame / fps
            duration = time - last_time
            if duration >= 0.3:  # avoid counting shot changing twice while fading in or fading out
                num_shots = num_shots + 1  # number of shots detected + 1
                last_time = time  # set last time for next loop
        s0 = s7  # set s0 as s7 for next loop when comparing with previous last frame

        # dissolve event detection
        if (abs(s1 - s7) >= 18):  # dissolve events happens when there is shot changing
            dissolveImg = cv2.add(frame1_old, frame7_old)  # overlap first and last frames
            dissolveImg = Image.fromarray(cv2.cvtColor(dissolveImg, cv2.COLOR_BGR2RGB))
            pd = imagehash.average_hash(dissolveImg)  # calculate hash value of overlapped image
            if (abs(pd - s4) <= 5):  # overlapped image should be similar to the image at central place in this loop
                feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7,
                                      blockSize=7)  # ShiTomasi Set parameters
                frame1_gray = cv2.cvtColor(frame1_old, cv2.COLOR_BGR2GRAY)
                frame7_gray = cv2.cvtColor(frame7_old, cv2.COLOR_BGR2GRAY)
                f1 = np.array(cv2.goodFeaturesToTrack(frame1_gray, mask=None, **feature_params))  # Get corner points
                f7 = np.array(cv2.goodFeaturesToTrack(frame7_gray, mask=None, **feature_params))  # Get corner points
                l1 = len(f1)
                l7 = len(f7)
                count = 0
                for i in range(l1):
                    for j in range(l7):
                        if ((f1[i] - f7[j]).any()):  # count total number of corner points that are not the same
                            count = count + 1
                if (count / (l1 * l7) >= 0.95 and frame7_gray.mean() >= 0.01 and frame1_gray.mean() >= 0.01):
                    num_Dissolve = num_Dissolve + 1  # dissolve events detected + 1
                    time = num_frame / fps  # time right now
                    print("detecting dissolve event at time ", time, "(seconds)")
                    Arr_Dsites.append(num_frame)  # store the number of frame at this dissolve site
    # total number of shots detected dose not include dissolve events,
    # who won't be detected here by shots changing detection
    num_shots = num_Dissolve + num_shots
    if (num_shots > 1):
        DIR = num_Dissolve / (num_shots - 1)  # Dissolve rate calculation
    else:
        DIR = 0.0
    # construct csv file to output all dissolve sites
    headers = ['number of frame', 'current time (s)']
    with open('Dissolve_Sites_All.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv',
              'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_Dsites)):
            d_f = Arr_Dsites[i]
            f_csv.writerow([d_f, d_f / fps])
    result_tag1.set("Dissolve Rate:")
    result1.set(DIR)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return DIR


# SECTION3 FUNCTIONS

def cameramotionintensityCORE(h, w, flow):
    countr = 0  # set 0 value of pixel summation on the right
    countl = 0  # set 0 value of pixel summation on the left
    countt = 0  # set 0 value of pixel summation on the top
    countb = 0  # set 0 value of pixel summation on the bottom
    Median_Velocity = 0.0
    # Create an array for storing all pixels' velocity square in one frame:
    Arr_VelocitySquare = arr.array('d', [])
    for i in range(h):
        a = (flow[i, w - 1][0]) ** 2 + (flow[i, w - 1][1]) ** 2  # pixel velocity square calculation
        countr = countr + a
        # summation of right most pixels' velocity square
        Arr_VelocitySquare.append(a)  # store the last duration
    if (countr / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
        for i in range(h):
            b = (flow[i, 0][0]) ** 2 + (flow[i, 0][1]) ** 2  # pixel velocity square calculation
            countl = countl + b
            # summation of left most pixels' velocity square
            Arr_VelocitySquare.append(b)  # store the last duration
        if (countl / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
            for i in range(w - 2):
                c = (flow[0, i + 1][0]) ** 2 + (flow[0, i + 1][1]) ** 2  # pixel velocity square calculation
                countt = countt + c
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(c)  # store the last duration
                d = (flow[h - 1, i + 1][0]) ** 2 + (
                    flow[h - 1, i + 1][1]) ** 2  # pixel velocity square calculation
                countb = countb + d
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(d)  # store the last duration
            SmalltoLarge_Arr = sorted(Arr_VelocitySquare)
            length = len(SmalltoLarge_Arr)
            # Calculate pixels' Median optical flow (velocity) depending on the odd/even value of its length
            if (length % 2) == 0:  # if even, use the mean value of the central two
                a = int((length / 2) - 1)
                b = int(length / 2)
                Median_Velocity = math.sqrt((SmalltoLarge_Arr[a] + SmalltoLarge_Arr[b]) / 2)
            else:
                Median_Velocity = math.sqrt(SmalltoLarge_Arr[round(length / 2) - 1])
    return Median_Velocity
def CameraMotionIntensity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    Arr_cmv = arr.array('d', [])  # create an array to store every frame's camera motion net velocity
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))  # image transformation
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        CMI_count = 0.0  # to add up MIn values of all shots
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                CMI_count = CMI_count + MIn_count / num_frame_local  # Update CMI_count to include the last shot's value
                CMI = CMI_count / num_shots  # after reading all frames return CMI value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                MIn = MIn_count / num_frame_local
                CMI_count = CMI_count + MIn
                # if shot changed, add last shot's MIn onto CMI_count
                MIn_count = 0.0  # update MIn_count to zero
                num_frame_local = 1  # update num_frame_local to zero
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
            ####### Core Calculation
            cm_v = cameramotionintensityCORE(h, w, flow)
            Arr_cmv.append(cm_v)  # store every frame's camera motion net velocity
            MIn_count = MIn_count + cm_v
        if (num_shots == 1):
            CMI = MIn_count / num_frame
            # when no shot change detected, update CMI as MIn_count / total number of frame
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        CMI_count = 0.0  # to add up MIn values of all shots
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
                ####### Core Calculation
                cm_v = cameramotionintensityCORE(h, w, flow)
                Arr_cmv.append(cm_v)  # store every frame's camera motion net velocity
                MIn_count = MIn_count + cm_v
            CMI_count = CMI_count + (MIn_count / Arr_shotlength_f[i])
            MIn_count = 0.0
        CMI = CMI_count / lenlst
    # construct csv file to output every frame's camera motion
    headers = ['number of frame', 'current time (s)', 'camera motion net velocity']
    with open('CameraMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmv)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmv[i]])
    result_tag1.set("Camera Motion Intensity (CMI):")
    result1.set(CMI)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return CMI
def CameraMotionIntensityVariance(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    Arr_cmv = arr.array('d', [])  # create an array to store every frame's camera motion net velocity
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_MIn = arr.array('d', [])  # Create an array for storing every value of MIn
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        CMI_count = 0.0  # to add up MIn values of all shots
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                MIn = MIn_count / num_frame_local  # calculate last shot's MIn
                Arr_MIn.append(MIn)  # store last shot's MIn value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                MIn = MIn_count / num_frame_local
                Arr_MIn.append(MIn)  # store MIn value from last shot
                CMI_count = CMI_count + MIn
                # if shot changed, add last shot's MIn onto CMI_count
                MIn_count = 0.0  # update MIn_count to zero
                num_frame_local = 1  # update num_frame_local to zero
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
            ###### Core Calculation
            cm_v = cameramotionintensityCORE(h, w, flow)
            Arr_cmv.append(cm_v)
            MIn_count = MIn_count + cm_v
        if (num_shots == 1):
            CIV = 0.0  # Camera Motion intensity Variance is zero
        else:
            MInSum = np.sum(Arr_MIn)  # Summation of all MIn value
            MInMean = MInSum / num_shots  # get mean MIn value
            VarSum = 0.0
            # calculate CIV when there are more than one shots
            for i in range(num_shots):
                VarSum = VarSum + (Arr_MIn[i] - MInMean) ** 2
            CIV = VarSum / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
                ####### Core Calculation
                cm_v = cameramotionintensityCORE(h, w, flow)
                Arr_cmv.append(cm_v)
                MIn_count = MIn_count + cm_v
            Arr_MIn.append(MIn_count / Arr_shotlength_f[i])  # store each shot's MIn value
        MInMean = np.sum(Arr_MIn) / lenlst
        VarSum = 0.0
        # calculate CIV when there are more than one shots
        for i in range (lenlst):
            VarSum = VarSum + (Arr_MIn[i] - MInMean) ** 2
        CIV = VarSum / lenlst
    # construct csv file to output every frame's camera motion
    headers = ['number of frame', 'current time (s)', 'camera motion net velocity']
    with open('CameraMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmv)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmv[i]])
    result_tag1.set("Camera Motion Intensity Variance(CIV):")
    result1.set(CIV)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return CIV

def cameramotioncomplexityCORE(h, w, flow, Arr_binh):
    Arr_VelocitySquare = arr.array('d', [])
    # Create an array for storing all pixels' velocity square in one frame
    Arr_Velocity3 = arr.array('d', [])
    # Create an array for storing all pixels' velocity square, x-velocity, and y-velocity
    countr = 0  # set 0 value of pixel summation on the right
    countl = 0  # set 0 value of pixel summation on the left
    countt = 0  # set 0 value of pixel summation on the top
    countb = 0  # set 0 value of pixel summation on the bottom
    for i in range(h):
        x = flow[i, w - 1][0]
        y = flow[i, w - 1][1]
        a = x ** 2 + y ** 2  # pixel velocity square calculation
        countr = countr + a
        # summation of right most pixels' velocity square
        Arr_VelocitySquare.append(a)  # store the last duration
        Arr_Velocity3.append(a)
        Arr_Velocity3.append(x)
        Arr_Velocity3.append(y)
    if (countr / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
        for i in range(h):
            x = flow[i, 0][0]
            y = flow[i, 0][1]
            b = x ** 2 + y ** 2  # pixel velocity square calculation
            countl = countl + b
            # summation of left most pixels' velocity square
            Arr_VelocitySquare.append(b)  # store the last duration
            Arr_Velocity3.append(b)
            Arr_Velocity3.append(x)
            Arr_Velocity3.append(y)
        if (countl / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
            for i in range(w - 2):
                x = flow[0, i + 1][0]
                y = flow[0, i + 1][1]
                c = x ** 2 + y ** 2  # pixel velocity square calculation
                countt = countt + c
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(c)  # store the last duration
                Arr_Velocity3.append(c)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
                x = flow[h - 1, i + 1][0]
                y = flow[h - 1, i + 1][1]
                d = x ** 2 + y ** 2  # pixel velocity square calculation
                countb = countb + d
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(d)  # store the last duration
                Arr_Velocity3.append(d)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
            SmalltoLarge_Arr = sorted(Arr_VelocitySquare)
            length = len(SmalltoLarge_Arr)
            # Calculate pixels' Median optical flow (velocity) depending on the odd/even value of its length
            if (length % 2) == 0:  # if even, use the mean value of the central two
                a = int((length / 2) - 1)
                b = int(length / 2)
                medianl = SmalltoLarge_Arr[a]
                medianr = SmalltoLarge_Arr[b]
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == medianl):
                        median_xl = Arr_Velocity3[1 + 3 * i]
                        median_yl = Arr_Velocity3[2 + 3 * i]
                    if (Arr_Velocity3[0 + 3 * i] == medianr):
                        median_xr = Arr_Velocity3[1 + 3 * i]
                        median_yr = Arr_Velocity3[2 + 3 * i]
                median_x = (median_xl + median_xr) / 2
                median_y = (median_yl + median_yr) / 2
            else:
                Median_Velocity = math.sqrt(SmalltoLarge_Arr[round(length / 2) - 1])
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == Median_Velocity):
                        median_x = Arr_Velocity3[1 + 3 * i]
                        median_y = Arr_Velocity3[2 + 3 * i]
            tanxy = median_y / median_x
            if (median_x >= 0):
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[0] = Arr_binh[0] + 1
                else:
                    if (median_y > 0):
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[1] = Arr_binh[1] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[7] = Arr_binh[7] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            else:
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[4] = Arr_binh[4] + 1
                else:
                    if (median_y > 0):
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[3] = Arr_binh[3] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[5] = Arr_binh[5] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
    return Arr_binh, median_x, median_y
def CameraMotionComplexity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_cmv = arr.array('d', [])  # create an array to store every frame's camera motion net velocity
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    CMC_count = 0.0  # to add up CMC value of every shot
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                # calculate camera motion complexity of last shot and add it up to CMC_count
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
            ###### Core Calculation
            Arr_binh, cm_x, cm_y = cameramotioncomplexityCORE(h, w, flow, Arr_binh)
            Arr_cmx.append(cm_x)
            Arr_cmy.append(cm_y)
            Arr_cmv.append(math.sqrt(cm_x ** 2 + cm_y ** 2))
        CMC = CMC_count / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
                ####### Core Calculation
                Arr_binh, cm_x, cm_y = cameramotioncomplexityCORE(h, w, flow, Arr_binh)
                Arr_cmx.append(cm_x)
                Arr_cmy.append(cm_y)
                Arr_cmv.append(math.sqrt(cm_x ** 2 + cm_y ** 2))
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    CMC_count = CMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to CMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
        CMC = CMC_count / lenlst
    # construct csv file to output every frame's camera motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y',
               'camera motion net velocity']
    with open('CameraMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_cmv[i]])
    result_tag1.set("Camera Motion Complexity (CMC):")
    result1.set(CMC)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return CMC
def CameraMotionComplexityVariance(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_cmv = arr.array('d', [])  # create an array to store every frame's camera motion net velocity
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    CMC_count = 0.0  # to add up CMC value of one shot
    Arr_CMC = arr.array('d', [])  # Create new array to store each shot's CMC value
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                Arr_CMC.append(CMC_count)  # store CMC value of the last shot
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                # calculate camera motion complexity of last shot and add it up to CMC_count
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
                Arr_CMC.append(CMC_count)  # store CMC value of the previous shot
                CMC_count = 0.0  # reset CMC_count value to zero for next shot's CMC value add up
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
            ###### Core Calculation
            Arr_binh, cm_x, cm_y = cameramotioncomplexityCORE(h, w, flow, Arr_binh)
            Arr_cmx.append(cm_x)
            Arr_cmy.append(cm_y)
            Arr_cmv.append(math.sqrt(cm_x ** 2 + cm_y ** 2))
        # when there is only one shot the CMC_count has not been calculated yet
        if (num_shots == 1):
            CCV = 0.0
        else:
            mean_CMC = np.sum(Arr_CMC) / num_shots
            VarSum = 0.0
            for i in range(num_shots):
                VarSum = VarSum + (Arr_CMC[i] - mean_CMC) ** 2
            CCV = VarSum / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
                ####### Core Calculation
                Arr_binh, cm_x, cm_y = cameramotioncomplexityCORE(h, w, flow, Arr_binh)
                Arr_cmx.append(cm_x)
                Arr_cmy.append(cm_y)
                Arr_cmv.append(math.sqrt(cm_x ** 2 + cm_y ** 2))
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    CMC_count = CMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to CMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
            Arr_CMC.append(CMC_count)  # store CMC value of the previous shot
            CMC_count = 0.0  # reset CMC_count value to zero for next shot's CMC value add up
        mean_CMC = np.sum(Arr_CMC) / lenlst
        VarSum = 0.0
        for i in range(lenlst):
            VarSum = VarSum + (Arr_CMC[i] - mean_CMC) ** 2
        CCV = VarSum / lenlst
    # construct csv file to output every frame's camera motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y',
               'camera motion net velocity']
    with open('CameraMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_cmv[i]])
    result_tag1.set("Camera Motion Complexity Variance (CCV):")
    result1.set(CCV)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return CCV

def cameramotionALLCORE(h, w, flow, Arr_binh):
    Median_Velocity = 0.0
    Arr_VelocitySquare = arr.array('d', [])
    # Create an array for storing all pixels' velocity square in one frame
    Arr_Velocity3 = arr.array('d', [])
    # Create an array for storing all pixels' velocity square, x-velocity, and y-velocity
    countr = 0  # set 0 value of pixel summation on the right
    countl = 0  # set 0 value of pixel summation on the left
    countt = 0  # set 0 value of pixel summation on the top
    countb = 0  # set 0 value of pixel summation on the bottom
    for i in range(h):
        x = flow[i, w - 1][0]
        y = flow[i, w - 1][1]
        a = x ** 2 + y ** 2  # pixel velocity square calculation
        countr = countr + a
        # summation of right most pixels' velocity square
        Arr_VelocitySquare.append(a)  # store the last duration
        Arr_Velocity3.append(a)
        Arr_Velocity3.append(x)
        Arr_Velocity3.append(y)
    if (countr / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
        for i in range(h):
            x = flow[i, 0][0]
            y = flow[i, 0][1]
            b = x ** 2 + y ** 2  # pixel velocity square calculation
            countl = countl + b
            # summation of left most pixels' velocity square
            Arr_VelocitySquare.append(b)  # store the last duration
            Arr_Velocity3.append(b)
            Arr_Velocity3.append(x)
            Arr_Velocity3.append(y)
        if (countl / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
            for i in range(w - 2):
                x = flow[0, i + 1][0]
                y = flow[0, i + 1][1]
                c = x ** 2 + y ** 2  # pixel velocity square calculation
                countt = countt + c
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(c)  # store the last duration
                Arr_Velocity3.append(c)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
                x = flow[h - 1, i + 1][0]
                y = flow[h - 1, i + 1][1]
                d = x ** 2 + y ** 2  # pixel velocity square calculation
                countb = countb + d
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(d)  # store the last duration
                Arr_Velocity3.append(d)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
            SmalltoLarge_Arr = sorted(Arr_VelocitySquare)
            length = len(SmalltoLarge_Arr)
            # Calculate pixels' Median optical flow (velocity) depending on the odd/even value of its length
            if (length % 2) == 0:  # if even, use the mean value of the central two
                a = int((length / 2) - 1)
                b = int(length / 2)
                medianl = SmalltoLarge_Arr[a]
                medianr = SmalltoLarge_Arr[b]
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == medianl):
                        median_xl = Arr_Velocity3[1 + 3 * i]
                        median_yl = Arr_Velocity3[2 + 3 * i]
                    if (Arr_Velocity3[0 + 3 * i] == medianr):
                        median_xr = Arr_Velocity3[1 + 3 * i]
                        median_yr = Arr_Velocity3[2 + 3 * i]
                median_x = (median_xl + median_xr) / 2
                median_y = (median_yl + median_yr) / 2
                Median_Velocity = math.sqrt((medianl + medianr) / 2)
            else:
                Median_Velocity = math.sqrt(SmalltoLarge_Arr[round(length / 2) - 1])
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == Median_Velocity):
                        median_x = Arr_Velocity3[1 + 3 * i]
                        median_y = Arr_Velocity3[2 + 3 * i]
            tanxy = median_y / median_x
            if (median_x >= 0):
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[0] = Arr_binh[0] + 1
                else:
                    if (median_y > 0):
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[1] = Arr_binh[1] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[7] = Arr_binh[7] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            else:
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[4] = Arr_binh[4] + 1
                else:
                    if (median_y > 0):
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[3] = Arr_binh[3] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[5] = Arr_binh[5] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
    return Arr_binh, Median_Velocity, median_x, median_y
def CameraMotionAll(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_cmv = arr.array('d', [])  # create an array to store every frame's camera motion net velocity
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    CMC_count = 0.0  # to add up CMC value of one shot
    Arr_CMC = arr.array('d', [])  # Create new array to store each shot's CMC value
    Arr_MIn = arr.array('d', [])  # Create an array for storing every value of MIn
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        CMI_count = 0.0  # to add up MIn values of all shots
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                MIn = MIn_count / num_frame_local  # calculate last shot's MIn value
                CMI_count = CMI_count + MIn
                CMI = CMI_count / num_shots  # after reading all frames return CMI value
                Arr_MIn.append(MIn)  # store last shot's MIn value
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                Arr_CMC.append(CMC_count)  # store CMC value of the last shot
                CMC = np.sum(Arr_CMC) / num_shots  # after reading all frames return CMC value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                MIn = MIn_count / num_frame_local
                Arr_MIn.append(MIn)  # store MIn value from last shot
                CMI_count = CMI_count + MIn
                # if shot changed, add last shot's MIn onto CMI_count
                MIn_count = 0.0  # update MIn_count to zero
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        CMC_count = CMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to CMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
                Arr_CMC.append(CMC_count)  # store CMC value of the previous shot
                CMC_count = 0.0  # reset CMC_count value to zero for next shot's CMC value add up
                num_frame_local = 1  # update num_frame_local to zero
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
            ####### Core Calculation
            Arr_binh, Median_Velocity, cm_x, cm_y = cameramotionALLCORE(h, w, flow, Arr_binh)
            MIn_count = MIn_count + Median_Velocity
            Arr_cmx.append(cm_x)
            Arr_cmy.append(cm_y)
            Arr_cmv.append(Median_Velocity)
        if (num_shots == 1):
            CIV = 0.0  # Camera Motion intensity Variance is zero
            CCV = 0.0  # Camera Motion Complexity Variance is zero
        else:
            MInSum = np.sum(Arr_MIn)  # Summation of all MIn value
            MInMean = MInSum / num_shots  # get mean MIn value
            VarSum = 0.0
            # calculate CIV when there are more than one shots
            for i in range(num_shots):
                VarSum = VarSum + (Arr_MIn[i] - MInMean) ** 2
            CIV = VarSum / num_shots
            CMCSum = np.sum(Arr_CMC)  # Summation of all CMC value
            CMCMean = CMCSum / num_shots  # get mean CMC value
            VarSum = 0.0
            # calculate CCV when there are more than one shots
            for i in range(num_shots):
                VarSum = VarSum + (Arr_CMC[i] - CMCMean) ** 2
            CCV = VarSum / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        MIn_count = 0.0  # to add up optical flow of each frame with camera motion
        CMI_count = 0.0  # to add up MIn values of all shots
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
                ####### Core Calculation
                Arr_binh, Median_Velocity, cm_x, cm_y = cameramotionALLCORE(h, w, flow, Arr_binh)
                MIn_count = MIn_count + Median_Velocity
                Arr_cmx.append(cm_x)
                Arr_cmy.append(cm_y)
                Arr_cmv.append(Median_Velocity)
                MIn_count = MIn_count + Median_Velocity
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    CMC_count = CMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to CMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
            Arr_CMC.append(CMC_count)  # store CMC value of the previous shot
            CMC_count = 0.0  # reset CMC_count value to zero for next shot's CMC value add up
            MIn = (MIn_count / Arr_shotlength_f[i])
            Arr_MIn.append(MIn)
            CMI_count = CMI_count + MIn
            MIn_count = 0.0
        CMI = CMI_count / lenlst
        CMC = np.sum(Arr_CMC) / lenlst  # after reading all frames return CMC value
        if (lenlst == 1):
            CIV = 0.0  # Camera Motion intensity Variance is zero
            CCV = 0.0  # Camera Motion Complexity Variance is zero
        else:
            MInSum = np.sum(Arr_MIn)  # Summation of all MIn value
            MInMean = MInSum / lenlst  # get mean MIn value
            VarSum = 0.0
            # calculate CIV when there are more than one shots
            for i in range(lenlst):
                VarSum = VarSum + (Arr_MIn[i] - MInMean) ** 2
            CIV = VarSum / lenlst
            VarSum = 0.0
            # calculate CCV when there are more than one shots
            for i in range(lenlst):
                VarSum = VarSum + (Arr_CMC[i] - CMC) ** 2
            CCV = VarSum / lenlst
    # construct csv file to output every frame's camera motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y',
               'camera motion net velocity']
    with open('CameraMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_cmv[i]])
    outa = tuple([CMI, CIV, CMC, CCV])
    result_tag1.set("Camera Motion Intensity (CMI), Camera Motion Intensity Variance (CIV), "
                    "Camera Motion Complexity (CMC), Camera Motion Complexity Variance (CCV):")
    result1.set(outa)
    result_tag2.set("csv file successfully constructed,")
    result2.set("Please check at current folder")
    # return CMI, CIV, CMC, CCV

def internalmotionintensityCORE(h, w, flow):
    median_x = 0  # set camera motion's x component to 0
    median_y = 0  # set camera motion's y component to 0
    Arr_VelocitySquare = arr.array('d', [])
    # Create an array for storing all pixels' velocity square in one frame
    Arr_Velocity3 = arr.array('d', [])
    # Create an array for storing all pixels' velocity square, x-velocity, and y-velocity
    countr = 0  # set 0 value of pixel summation on the right
    countl = 0  # set 0 value of pixel summation on the left
    countt = 0  # set 0 value of pixel summation on the top
    countb = 0  # set 0 value of pixel summation on the bottom
    for i in range(h):
        x = flow[i, w - 1][0]
        y = flow[i, w - 1][1]
        a = x ** 2 + y ** 2  # pixel velocity square calculation
        countr = countr + a
        # summation of right most pixels' velocity square
        Arr_VelocitySquare.append(a)  # store the last duration
        Arr_Velocity3.append(a)
        Arr_Velocity3.append(x)
        Arr_Velocity3.append(y)
    if (countr / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
        for i in range(h):
            x = flow[i, 0][0]
            y = flow[i, 0][1]
            b = x ** 2 + y ** 2  # pixel velocity square calculation
            countl = countl + b
            # summation of left most pixels' velocity square
            Arr_VelocitySquare.append(b)  # store the last duration
            Arr_Velocity3.append(b)
            Arr_Velocity3.append(x)
            Arr_Velocity3.append(y)
        if (countl / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
            for i in range(w - 2):
                x = flow[0, i + 1][0]
                y = flow[0, i + 1][1]
                c = x ** 2 + y ** 2  # pixel velocity square calculation
                countt = countt + c
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(c)  # store the last duration
                Arr_Velocity3.append(c)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
                x = flow[h - 1, i + 1][0]
                y = flow[h - 1, i + 1][1]
                d = x ** 2 + y ** 2  # pixel velocity square calculation
                countb = countb + d
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(d)  # store the last duration
                Arr_Velocity3.append(d)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
            SmalltoLarge_Arr = sorted(Arr_VelocitySquare)
            length = len(SmalltoLarge_Arr)
            # Calculate pixels' Median optical flow (velocity) depending on the odd/even value of its length
            if (length % 2) == 0:  # if even, use the mean value of the central two
                a = int((length / 2) - 1)
                b = int(length / 2)
                medianl = SmalltoLarge_Arr[a]
                medianr = SmalltoLarge_Arr[b]
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == medianl):
                        median_xl = Arr_Velocity3[1 + 3 * i]
                        median_yl = Arr_Velocity3[2 + 3 * i]
                    if (Arr_Velocity3[0 + 3 * i] == medianr):
                        median_xr = Arr_Velocity3[1 + 3 * i]
                        median_yr = Arr_Velocity3[2 + 3 * i]
                median_x = (median_xl + median_xr) / 2
                median_y = (median_yl + median_yr) / 2
            else:
                Median_Velocity = math.sqrt(SmalltoLarge_Arr[round(length / 2) - 1])
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == Median_Velocity):
                        median_x = Arr_Velocity3[1 + 3 * i]
                        median_y = Arr_Velocity3[2 + 3 * i]
    return median_x, median_y
def InternalMotionIntensity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    IMI_count = 0.0  # to add up internal motion intensity of each shot
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_imx = arr.array('d', [])  # create an array to store every frame's internal motion x component
    Arr_imy = arr.array('d', [])  # create an array to store every frame's internal motion y component
    Arr_imv = arr.array('d', [])  # create an array to store every frame's net internal motion velocity
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                num_frame_local = 1  # reset num_frame_local to zero
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            ###### Core Calculation
            cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
            Arr_cmx.append(cm_x)  # store every frame's camera motion x component
            Arr_cmy.append(cm_y)  # store every frame's camera motion y component
            # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
            totalmotion_count = np.sum(flow, axis=0)
            totalmotion_count = np.sum(totalmotion_count, axis=0)
            internalmotion_x = totalmotion_count[0]/(h * w) - cm_x  # deduct camera motion x component from flow vector
            internalmotion_y = totalmotion_count[1]/(h * w) - cm_y  # deduct camera motion y component from flow vector
            Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
            Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
            # calculate total motion:
            totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
            Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
            IMI_count = IMI_count + totalmotion_velocity  # add up this frame's total internal motion
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
        IMI = IMI_count / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                ###### Core Calculation
                cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
                Arr_cmx.append(cm_x)  # store every frame's camera motion x component
                Arr_cmy.append(cm_y)  # store every frame's camera motion y component
                if (cm_x != 0 and cm_y != 0):
                    for m in range(h):
                        for n in range(w):
                            flow[m, n][0] = flow[m, n][0] - cm_x  # deduct camera motion x component from flow vector
                            flow[m, n][1] = flow[m, n][1] - cm_y  # deduct camera motion y component from flow vector
                # to add up total motion in each frame (sum every column/row and then sum every column/row):
                totalmotion_count = np.sum(flow, axis=0)
                totalmotion_count = np.sum(totalmotion_count, axis=0)
                internalmotion_x = totalmotion_count[0] / (h * w)
                internalmotion_y = totalmotion_count[1] / (h * w)
                Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
                Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
                # calculate total motion:
                totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
                Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
                IMI_count = IMI_count + totalmotion_velocity
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
        IMI = IMI_count / lenlst
    # construct csv file to output every frame's camera motion & internal motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y', 'internal motion mean_x',
               'internal motion mean_y', 'internal motion velocity']
    with open('InternalMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_imx[i], Arr_imy[i], Arr_imv[i]])
    result_tag1.set("Internal Motion Intensity (IMI):")
    result1.set(IMI)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder!")
    # return IMI
def InternalMotionIntensityVariance(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    IMI_count  = 0.0  # to add up every shot's internal  motion
    Arr_IIV = arr.array('d', [])  # Create a new array to store every shot's IMI
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_imx = arr.array('d', [])  # create an array to store every frame's internal motion x component
    Arr_imy = arr.array('d', [])  # create an array to store every frame's internal motion y component
    Arr_imv = arr.array('d', [])  # create an array to store every frame's net internal motion velocity
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                Arr_IIV.append(IMI_count / num_frame_local)  # store the last shot's IMI value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                Arr_IIV.append(IMI_count / num_frame_local)  # store last shot's IMI value
                IMI_count = 0.0  # reset IMI_count to zero to count internal motion of next shot
                num_frame_local = 1  # reset num_frame_local to zero
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            ###### Core Calculation
            cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
            Arr_cmx.append(cm_x)  # store every frame's camera motion x component
            Arr_cmy.append(cm_y)  # store every frame's camera motion y component
            # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
            totalmotion_count = np.sum(flow, axis=0)
            totalmotion_count = np.sum(totalmotion_count, axis=0)
            internalmotion_x = totalmotion_count[0]/(h * w) - cm_x  # deduct camera motion x component from flow vector
            internalmotion_y = totalmotion_count[1]/(h * w) - cm_y  # deduct camera motion y component from flow vector
            Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
            Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
            # calculate total motion:
            totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
            Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
            IMI_count = IMI_count + totalmotion_velocity  # add up every frame's internal motion
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
        # internal motion intensity variance calculation:
        if (num_shots == 1):  # when MIn and MVn are not calculated yet
            IIV = 0.0
        else:
            IIV_mean = np.sum(Arr_IIV) / num_shots
            IIV_count = 0.0
            for i in range(num_shots):
                IIV_count = IIV_count + (Arr_IIV[i] - IIV_mean) ** 2
            IIV = IIV_count / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                ###### Core Calculation
                cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
                Arr_cmx.append(cm_x)  # store every frame's camera motion x component
                Arr_cmy.append(cm_y)  # store every frame's camera motion y component
                if (cm_x != 0 and cm_y != 0):
                    for m in range(h):
                        for n in range(w):
                            flow[m, n][0] = flow[m, n][0] - cm_x  # deduct camera motion x component from flow vector
                            flow[m, n][1] = flow[m, n][1] - cm_y  # deduct camera motion y component from flow vector
                # to add up total motion in each frame (sum every column/row and then sum every column/row):
                totalmotion_count = np.sum(flow, axis=0)
                totalmotion_count = np.sum(totalmotion_count, axis=0)
                internalmotion_x = totalmotion_count[0] / (h * w)
                internalmotion_y = totalmotion_count[1] / (h * w)
                Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
                Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
                # calculate total motion:
                totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
                Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
                IMI_count = IMI_count + totalmotion_velocity
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
            Arr_IIV.append(IMI_count / Arr_shotlength_f[i])  # store last shot's IMI value
            IMI_count = 0.0  # reset IMI_count to zero to count internal motion of next shot
        # internal motion intensity variance calculation:
        IIV_mean = np.sum(Arr_IIV) / lenlst
        IIV_count = 0.0
        for i in range(lenlst):
            IIV_count = IIV_count + (Arr_IIV[i] - IIV_mean) ** 2
        IIV = IIV_count / lenlst
    # construct csv file to output every frame's camera motion & internal motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y', 'internal motion mean_x',
               'internal motion mean_y', 'internal motion velocity']
    with open('InternalMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_imx[i], Arr_imy[i], Arr_imv[i]])
    result_tag1.set("Internal Motion Intensity Variance (IIV):")
    result1.set(IIV)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder")
    # return IIV
def InternalMotionComplexity(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    IMC_count = 0.0  # to add up IMC value of every shot
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_imx = arr.array('d', [])  # create an array to store every frame's internal motion x component
    Arr_imy = arr.array('d', [])  # create an array to store every frame's internal motion y component
    Arr_imv = arr.array('d', [])  # create an array to store every frame's net internal motion velocity
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                # calculate internal motion complexity of the last shot and add it up to IMC_count:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                # calculate internal motion complexity of last shot and add it up to IMC_count
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            ###### Core Calculation
            cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
            Arr_cmx.append(cm_x)  # store every frame's camera motion x component
            Arr_cmy.append(cm_y)  # store every frame's camera motion y component
            # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
            totalmotion_count = np.sum(flow, axis=0)
            totalmotion_count = np.sum(totalmotion_count, axis=0)
            internalmotion_x = totalmotion_count[0]/(h * w) - cm_x  # deduct camera motion x component from flow vector
            internalmotion_y = totalmotion_count[1]/(h * w) - cm_y  # deduct camera motion y component from flow vector
            Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
            Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
            # calculate total motion:
            totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
            Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
            # different angle occurrence add up
            tanxy = (cm_y - internalmotion_y) / (cm_x - internalmotion_x)
            if (cm_x >= 0):
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[0] = Arr_binh[0] + 1
                else:
                    if (cm_y > 0):
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[1] = Arr_binh[1] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[7] = Arr_binh[7] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            else:
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[4] = Arr_binh[4] + 1
                else:
                    if (cm_y > 0):
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[3] = Arr_binh[3] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[5] = Arr_binh[5] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
        IMC = IMC_count / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                ###### Core Calculation
                cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
                Arr_cmx.append(cm_x)  # store every frame's camera motion x component
                Arr_cmy.append(cm_y)  # store every frame's camera motion y component
                if (cm_x != 0 and cm_y != 0):
                    for m in range(h):
                        for n in range(w):
                            flow[m, n][0] = flow[m, n][0] - cm_x  # deduct camera motion x component from flow vector
                            flow[m, n][1] = flow[m, n][1] - cm_y  # deduct camera motion y component from flow vector
                # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
                totalmotion_count = np.sum(flow, axis=0)
                totalmotion_count = np.sum(totalmotion_count, axis=0)
                internalmotion_x = totalmotion_count[0] / (h * w)
                internalmotion_y = totalmotion_count[1] / (h * w)
                Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
                Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
                # calculate total motion:
                totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
                Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
                # different angle occurrence add up
                tanxy = (cm_y - internalmotion_y) / (cm_x - internalmotion_x)
                if (cm_x >= 0):
                    if (abs(tanxy) <= 0.4142):
                        Arr_binh[0] = Arr_binh[0] + 1
                    else:
                        if (cm_y > 0):
                            if (2.4142 >= tanxy > 0.4142):
                                Arr_binh[1] = Arr_binh[1] + 1
                            else:
                                Arr_binh[2] = Arr_binh[2] + 1
                        else:
                            if (-0.4142 >= tanxy >= -2.4142):
                                Arr_binh[7] = Arr_binh[7] + 1
                            else:
                                Arr_binh[6] = Arr_binh[6] + 1
                else:
                    if (abs(tanxy) <= 0.4142):
                        Arr_binh[4] = Arr_binh[4] + 1
                    else:
                        if (cm_y > 0):
                            if (-0.4142 >= tanxy >= -2.4142):
                                Arr_binh[3] = Arr_binh[3] + 1
                            else:
                                Arr_binh[2] = Arr_binh[2] + 1
                        else:
                            if (2.4142 >= tanxy > 0.4142):
                                Arr_binh[5] = Arr_binh[5] + 1
                            else:
                                Arr_binh[6] = Arr_binh[6] + 1
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    IMC_count = IMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to IMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
        IMC = IMC_count / lenlst
    # construct csv file to output every frame's camera motion & internal motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y', 'internal motion mean_x',
               'internal motion mean_y', 'internal motion velocity']
    with open('InternalMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_imx[i], Arr_imy[i], Arr_imv[i]])
    result_tag1.set("Internal Motion Complexity (IMC):")
    result1.set(IMC)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder")
    # return IMC
def InternalMotionComplexityVariance(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    IMC_count = 0.0  # to add up IMC value of every shot
    Arr_shotIMC = arr.array('d', [])  # Create a new array to store every shot's IMC value
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_imx = arr.array('d', [])  # create an array to store every frame's internal motion x component
    Arr_imy = arr.array('d', [])  # create an array to store every frame's internal motion y component
    Arr_imv = arr.array('d', [])  # create an array to store every frame's net internal motion velocity
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                # calculate internal motion complexity of the last shot and add it up to IMC_count:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                Arr_shotIMC.append(IMC_count)  # store the last shot's IMC value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                # calculate internal motion complexity of last shot and add it up to IMC_count
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
                Arr_shotIMC.append(IMC_count)  # store last shot's IMC value
                IMC_count = 0.0  # reset IMC_count to zero to count next shot's IMC value
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            ###### Core Calculation
            cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
            Arr_cmx.append(cm_x)  # store every frame's camera motion x component
            Arr_cmy.append(cm_y)  # store every frame's camera motion y component
            # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
            totalmotion_count = np.sum(flow, axis=0)
            totalmotion_count = np.sum(totalmotion_count, axis=0)
            internalmotion_x = totalmotion_count[0]/(h * w) - cm_x  # deduct camera motion x component from flow vector
            internalmotion_y = totalmotion_count[1]/(h * w) - cm_y  # deduct camera motion y component from flow vector
            Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
            Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
            # calculate total motion:
            totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
            Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
            # different angle occurrence add up
            tanxy = (cm_y - internalmotion_y) / (cm_x - internalmotion_x)
            if (cm_x >= 0):
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[0] = Arr_binh[0] + 1
                else:
                    if (cm_y > 0):
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[1] = Arr_binh[1] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[7] = Arr_binh[7] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            else:
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[4] = Arr_binh[4] + 1
                else:
                    if (cm_y > 0):
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[3] = Arr_binh[3] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[5] = Arr_binh[5] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
        # when there is only one shot the CMC_count has not been calculated yet
        if (num_shots == 1):
            ICV = 0.0
        else:
            mean_shotIMC = np.sum(Arr_shotIMC) / num_shots  # mean IMC value of all shots
            Varsum = 0.0
            for i in range(len(Arr_shotIMC)):
                Varsum = Varsum + (Arr_shotIMC[i] - mean_shotIMC) ** 2
            ICV = Varsum / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                ###### Core Calculation
                cm_x, cm_y = internalmotionintensityCORE(h, w, flow)  # calculate camera motion x & y components
                Arr_cmx.append(cm_x)  # store every frame's camera motion x component
                Arr_cmy.append(cm_y)  # store every frame's camera motion y component
                if (cm_x != 0 and cm_y != 0):
                    for m in range(h):
                        for n in range(w):
                            flow[m, n][0] = flow[m, n][0] - cm_x  # deduct camera motion x component from flow vector
                            flow[m, n][1] = flow[m, n][1] - cm_y  # deduct camera motion y component from flow vector
                # to add up total motion (optical flow) in each frame(sum every column/row and then sum every column/row):
                totalmotion_count = np.sum(flow, axis=0)
                totalmotion_count = np.sum(totalmotion_count, axis=0)
                internalmotion_x = totalmotion_count[0] / (h * w)
                internalmotion_y = totalmotion_count[1] / (h * w)
                Arr_imx.append(internalmotion_x)  # store every frame's internal motion x component
                Arr_imy.append(internalmotion_y)  # store every frame's internal motion y component
                # calculate total motion:
                totalmotion_velocity = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)
                Arr_imv.append(totalmotion_velocity)  # store every frame's net internal motion velocity
                # different angle occurrence add up
                tanxy = (cm_y - internalmotion_y) / (cm_x - internalmotion_x)
                if (cm_x >= 0):
                    if (abs(tanxy) <= 0.4142):
                        Arr_binh[0] = Arr_binh[0] + 1
                    else:
                        if (cm_y > 0):
                            if (2.4142 >= tanxy > 0.4142):
                                Arr_binh[1] = Arr_binh[1] + 1
                            else:
                                Arr_binh[2] = Arr_binh[2] + 1
                        else:
                            if (-0.4142 >= tanxy >= -2.4142):
                                Arr_binh[7] = Arr_binh[7] + 1
                            else:
                                Arr_binh[6] = Arr_binh[6] + 1
                else:
                    if (abs(tanxy) <= 0.4142):
                        Arr_binh[4] = Arr_binh[4] + 1
                    else:
                        if (cm_y > 0):
                            if (-0.4142 >= tanxy >= -2.4142):
                                Arr_binh[3] = Arr_binh[3] + 1
                            else:
                                Arr_binh[2] = Arr_binh[2] + 1
                        else:
                            if (2.4142 >= tanxy > 0.4142):
                                Arr_binh[5] = Arr_binh[5] + 1
                            else:
                                Arr_binh[6] = Arr_binh[6] + 1
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    IMC_count = IMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to IMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
            Arr_shotIMC.append(IMC_count)  # store last shot's IMC value
            IMC_count = 0.0  # reset IMC_count to zero to count next shot's IMC value
        mean_shotIMC = np.sum(Arr_shotIMC) / lenlst  # mean IMC value of all shots
        Varsum = 0.0
        for i in range(len(Arr_shotIMC)):
            Varsum = Varsum + (Arr_shotIMC[i] - mean_shotIMC) ** 2
        ICV = Varsum / lenlst
    # construct csv file to output every frame's camera motion & internal motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y', 'internal motion mean_x',
               'internal motion mean_y', 'internal motion velocity']
    with open('InternalMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_imx[i], Arr_imy[i], Arr_imv[i]])
    result_tag1.set("Internal Motion Complexity Variance (ICV):")
    result1.set(ICV)
    result2.set("csv file successfully constructed,")
    result_tag2.set("Please check at current folder")
    # return ICV

def internalmotionAllCORE(h, w, flow, totalmotion_x, totalmotion_y, Arr_binh):
    median_x = 0.0  # set initial camera motion x component to be zero
    median_y = 0.0  # set initial camera motion y component to be zero
    Median_Velocity = 0.0  # set initial camera motion to be zero
    # Create an array for storing all pixels' velocity square in one frame:
    Arr_VelocitySquare = arr.array('d', [])
    # Create an array for storing all pixels' velocity square, x-velocity, and y-velocity:
    Arr_Velocity3 = arr.array('d', [])
    countr = 0  # set 0 value of pixel summation on the right
    countl = 0  # set 0 value of pixel summation on the left
    countt = 0  # set 0 value of pixel summation on the top
    countb = 0  # set 0 value of pixel summation on the bottom
    for i in range(h):
        x = flow[i, w - 1][0]
        y = flow[i, w - 1][1]
        a = x ** 2 + y ** 2  # pixel velocity square calculation
        countr = countr + a
        # summation of right most pixels' velocity square
        Arr_VelocitySquare.append(a)  # store the last duration
        Arr_Velocity3.append(a)
        Arr_Velocity3.append(x)
        Arr_Velocity3.append(y)
    if (countr / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
        for i in range(h):
            x = flow[i, 0][0]
            y = flow[i, 0][1]
            b = x ** 2 + y ** 2  # pixel velocity square calculation
            countl = countl + b
            # summation of left most pixels' velocity square
            Arr_VelocitySquare.append(b)  # store the last duration
            Arr_Velocity3.append(b)
            Arr_Velocity3.append(x)
            Arr_Velocity3.append(y)
        if (countl / h >= 0.8):  # detecting camera motion if average pixel velocity >= 0.8
            for i in range(w - 2):
                x = flow[0, i + 1][0]
                y = flow[0, i + 1][1]
                c = x ** 2 + y ** 2  # pixel velocity square calculation
                countt = countt + c
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(c)  # store the last duration
                Arr_Velocity3.append(c)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
                x = flow[h - 1, i + 1][0]
                y = flow[h - 1, i + 1][1]
                d = x ** 2 + y ** 2  # pixel velocity square calculation
                countb = countb + d
                # summation of top most pixels' velocity square
                Arr_VelocitySquare.append(d)  # store the last duration
                Arr_Velocity3.append(d)
                Arr_Velocity3.append(x)
                Arr_Velocity3.append(y)
            SmalltoLarge_Arr = sorted(Arr_VelocitySquare)
            length = len(SmalltoLarge_Arr)
            # Calculate pixels' Median optical flow (velocity) depending on the odd/even value of its length
            if (length % 2) == 0:  # if even, use the mean value of the central two
                a = int((length / 2) - 1)
                b = int(length / 2)
                medianl = SmalltoLarge_Arr[a]
                medianr = SmalltoLarge_Arr[b]
                Median_Velocity = math.sqrt((medianl + medianr) / 2)
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == medianl):
                        median_xl = Arr_Velocity3[1 + 3 * i]
                        median_yl = Arr_Velocity3[2 + 3 * i]
                    if (Arr_Velocity3[0 + 3 * i] == medianr):
                        median_xr = Arr_Velocity3[1 + 3 * i]
                        median_yr = Arr_Velocity3[2 + 3 * i]
                median_x = (median_xl + median_xr) / 2
                median_y = (median_yl + median_yr) / 2
            else:
                Median_Velocity = math.sqrt(SmalltoLarge_Arr[round(length / 2) - 1])
                for i in range(length):
                    if (Arr_Velocity3[0 + 3 * i] == Median_Velocity):
                        median_x = Arr_Velocity3[1 + 3 * i]
                        median_y = Arr_Velocity3[2 + 3 * i]
            tanxy = (median_y - totalmotion_y) / (median_x - totalmotion_x)
            if (median_x >= 0):
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[0] = Arr_binh[0] + 1
                else:
                    if (median_y > 0):
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[1] = Arr_binh[1] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[7] = Arr_binh[7] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
            else:
                if (abs(tanxy) <= 0.4142):
                    Arr_binh[4] = Arr_binh[4] + 1
                else:
                    if (median_y > 0):
                        if (-0.4142 >= tanxy >= -2.4142):
                            Arr_binh[3] = Arr_binh[3] + 1
                        else:
                            Arr_binh[2] = Arr_binh[2] + 1
                    else:
                        if (2.4142 >= tanxy > 0.4142):
                            Arr_binh[5] = Arr_binh[5] + 1
                        else:
                            Arr_binh[6] = Arr_binh[6] + 1
    return Arr_binh, Median_Velocity, median_x, median_y
def InternalMotionAll(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    string = shot_changing_entry.get()
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    Arr_binh = arr.array('d', [])  # to store bin height of different angles with bin width of 45 degree
    zero = 0
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    Arr_binh.append(zero)
    MIn_count = 0.0  # to add up optical flow of each frame with camera motion
    MVn_count = 0.0  # to add up total motion of each shot
    IMC_count = 0.0  # to add up IMC value of every shot
    Arr_IIV = arr.array('d', [])  # Create a new array to store every shot's IMI
    Arr_shotIMC = arr.array('d', [])  # Create a new array to store every shot's IMC value
    Arr_cmx = arr.array('d', [])  # create an array to store every frame's camera motion x component
    Arr_cmy = arr.array('d', [])  # create an array to store every frame's camera motion y component
    Arr_imx = arr.array('d', [])  # create an array to store every frame's internal motion x component
    Arr_imy = arr.array('d', [])  # create an array to store every frame's internal motion y component
    Arr_imv = arr.array('d', [])  # create an array to store every frame's net internal motion velocity
    if (len(string) == 0):
        list = []  # create an empty list to store every last second's hash change values
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, old_frame = cap.read()  # get first frame
        num_frame_local = 1  # count number of frame within one shot
        num_frame = 1
        num_shots = 1
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        p0 = imagehash.average_hash(old_PIL, hash_size=9)  # hash value of old image
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        while (1):
            ret, new_frame = cap.read()  # read new frame
            if ret == False:
                MIn = MIn_count / num_frame_local  # the last shot's MIn
                MVn = MVn_count / num_frame_local  # the last shot's MVn
                Arr_IIV.append(MVn - MIn)  # store the last shot's IMI value
                # calculate internal motion complexity of the last shot and add it up to IMC_count:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                Arr_shotIMC.append(IMC_count)  # store the last shot's IMC value
                break
            num_frame = num_frame + 1
            new_PIL = Image.fromarray(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            p1 = imagehash.average_hash(new_PIL, hash_size=9)  # hash value of new image
            delta_y = abs(p0 - p1)
            lenlist = len(list)
            if lenlist == 0:
                hc_mean = 0.0
            else:
                if lenlist <= (fps / 4):
                    hc_mean = sum(list) / lenlist
                else:
                    list.pop(0)  # delete one element that is prior to the last second
                    hc_mean = sum(list) / (lenlist - 1)
            list.append(delta_y)  # store hash change value just calculated
            if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean):
                num_shots = num_shots + 1  # one more shot detected
                MIn = MIn_count / num_frame_local  # last shot's MIn
                MVn = MVn_count / num_frame_local  # last shot's MVn
                Arr_IIV.append(MVn - MIn)  # store last shot's IMI value
                MIn_count = 0.0  # reset MIn_count to zero to count camera motion of next shot
                MVn_count = 0.0  # reset MVn_count to zero to count total motion of next shot
                num_frame_local = 1  # update num_frame_local to zero
                # calculate internal motion complexity of last shot and add it up to IMC_count:
                for i in range(8):
                    if (Arr_binh[i] != 0):  # zero value of bin height has no meaning when taking log10
                        IMC_count = IMC_count + Arr_binh[i] * math.log10(Arr_binh[i])  # add up complexity to IMC_count
                        # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                        Arr_binh[i] = 0
                Arr_shotIMC.append(IMC_count)  # store last shot's IMC value
                IMC_count = 0.0  # reset IMC_count to zero to count next shot's IMC value
            else:
                num_frame_local = num_frame_local + 1  # num_frame_local plus one
            gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
            # optical flow calculation:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
            # total motion add up, using the flow matrix
            totalmotion_count = np.sum(flow, axis=0)  # summation of every column
            totalmotion_count = np.sum(totalmotion_count, axis=0)  # summation of every row
            totalmotion_x = totalmotion_count[0] / (h * w)  # total motion along x axis
            totalmotion_y = totalmotion_count[1] / (h * w)  # total motion along y axis
            ####### Core Calculation
            Arr_binh, Median_Velocity, cm_x, cm_y = internalmotionAllCORE(h, w, flow, totalmotion_x, totalmotion_y,
                                                                          Arr_binh)
            internalmotion_x = totalmotion_x - cm_x  # mean internal motion x component
            internalmotion_y = totalmotion_y - cm_y  # mean internal motion y component
            internalmotion = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)  # mean internal motion
            Arr_cmx.append(cm_x)
            Arr_cmy.append(cm_y)
            Arr_imx.append(internalmotion_x)
            Arr_imy.append(internalmotion_y)
            Arr_imv.append(internalmotion)
            MIn_count = MIn_count + Median_Velocity  # add up camera motion within one shot
            MVn_count = MVn_count + internalmotion  # add up total motion within one shot
            # for next iteration we update:
            gray1 = gray2  # set old frame as the frame just read for going through the loop again
            p0 = p1  # update the hash value just calculated to old hash value
        # INTERNAL MOTION INTENSITY #:
        IMI = np.sum(Arr_IIV) / num_shots  # also the mean for variance calculation
        # INTERNAL MOTION INTENSITY VARIANCE #:
        IIV_count = 0.0
        for i in range(len(Arr_IIV)):
            IIV_count = IIV_count + (Arr_IIV[i] - IMI) ** 2
        IIV = IIV_count / num_shots
        # INTERNAL MOTION COMPLEXITY #:
        IMC = np.sum(Arr_shotIMC) / num_shots
        # INTERNAL MOTION COMPLEXITY VARIANCE #:
        ICV_count = 0.0
        for i in range(len(Arr_shotIMC)):
            ICV_count = ICV_count + (Arr_shotIMC[i] - IMC) ** 2
        ICV = ICV_count / num_shots
    else:
        lst = arr.array('f', [float(x) for x in string.split(',')])
        lenlst = len(lst)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        Arr_shotlength_f = arr.array('d', [])  # create new array to store each shot's number of frame
        for i in range(lenlst):
            if (i != 0):
                Arr_shotlength_f.append(round((lst[i] - lst[i - 1]) * fps))
            else:
                Arr_shotlength_f.append(round((lst[i]) * fps))
        ret, old_frame = cap.read()  # get first frame
        gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        h = old_frame.shape[0]  # read frame height
        w = old_frame.shape[1]  # read frame weight
        for i in range(lenlst):
            j = 1
            while (j <= Arr_shotlength_f[i]):
                ret, new_frame = cap.read()  # read new frame
                if ret == False:
                    break
                j = j + 1
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert new frame to GRAY style
                # optical flow calculation:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)
                # total motion add up, using the flow matrix
                totalmotion_count = np.sum(flow, axis=0)  # summation of every column
                totalmotion_count = np.sum(totalmotion_count, axis=0)  # summation of every row
                totalmotion_x = totalmotion_count[0] / (h * w)  # total motion along x axis
                totalmotion_y = totalmotion_count[1] / (h * w)  # total motion along y axis
                ####### Core Calculation
                Arr_binh, Median_Velocity, cm_x, cm_y = internalmotionAllCORE(h, w, flow, totalmotion_x, totalmotion_y,
                                                                              Arr_binh)
                internalmotion_x = totalmotion_x - cm_x  # mean internal motion x component
                internalmotion_y = totalmotion_y - cm_y  # mean internal motion y component
                internalmotion = math.sqrt(internalmotion_x ** 2 + internalmotion_y ** 2)  # mean internal motion
                Arr_cmx.append(cm_x)
                Arr_cmy.append(cm_y)
                Arr_imx.append(internalmotion_x)
                Arr_imy.append(internalmotion_y)
                Arr_imv.append(internalmotion)
                MIn_count = MIn_count + Median_Velocity  # add up camera motion within one shot
                MVn_count = MVn_count + internalmotion  # add up total motion within one shot
                # for next iteration we update:
                gray1 = gray2  # set old frame as the frame just read for going through the loop again
            MIn = MIn_count / Arr_shotlength_f[i]
            MVn = MVn_count / Arr_shotlength_f[i]
            Arr_IIV.append(MVn - MIn)
            MIn_count = 0.0  # reset MIn_count to zero to count camera motion of next shot
            MVn_count = 0.0  # reset MVn_count to zero to count total motion of next shot
            for k in range(8):
                if (Arr_binh[k] != 0):  # zero value of bin height has no meaning when taking log10
                    IMC_count = IMC_count + Arr_binh[k] * math.log10(Arr_binh[k])  # add up complexity to IMC_count
                    # reset every non-zero Arr-bin[i] to zero so all value of this array is reset to zero:
                    Arr_binh[k] = 0
            Arr_shotIMC.append(IMC_count)  # store last shot's IMC value
            IMC_count = 0.0  # reset IMC_count to zero to count next shot's IMC value
        # INTERNAL MOTION INTENSITY #:
        IMI = np.sum(Arr_IIV) / lenlst  # also the mean for variance calculation
        # INTERNAL MOTION INTENSITY VARIANCE #:
        IIV_count = 0.0
        for i in range(len(Arr_IIV)):
            IIV_count = IIV_count + (Arr_IIV[i] - IMI) ** 2
        IIV = IIV_count / lenlst
        # INTERNAL MOTION COMPLEXITY #:
        IMC = np.sum(Arr_shotIMC) / lenlst
        # INTERNAL MOTION COMPLEXITY VARIANCE #:
        ICV_count = 0.0
        for i in range(len(Arr_shotIMC)):
            ICV_count = ICV_count + (Arr_shotIMC[i] - IMC) ** 2
        ICV = ICV_count / lenlst
    # construct csv file to output every frame's camera motion & internal motion
    headers = ['number of frame', 'current time (s)', 'camera motion_x', 'camera motion_y', 'internal motion mean_x',
               'internal motion mean_y', 'internal motion velocity']
    with open('InternalMotion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_cmx)):
            f_csv.writerow([i + 2, (i+2) / fps, Arr_cmx[i], Arr_cmy[i], Arr_imx[i], Arr_imy[i], Arr_imv[i]])
    outa = tuple([IMI, IIV, IMC, ICV])
    result_tag1.set("Internal Motion Intensity (IMI), Internal Motion Intensity Variance (IIV):")
    result1.set(outa)
    result_tag2.set("csv file successfully constructed,")
    result2.set("Please check at current folder")
    # return IMI, IIV, IMC, ICV


# SECTION3 OPTICALFLOW

def OpticalFlowHornSchunck(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    step = 10
    Arr_x = arr.array('d', [])  # create an array to store every frame's total motion x component
    Arr_y = arr.array('d', [])  # create an array to store every frame's total motion y component
    Arr_net = arr.array('d', [])  # create an array to store every frame's total motion net velocity
    ret, old_frame = cap.read()  # get first frame
    num_frame = 1  # count number of frame so far
    gray1 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    h = old_frame.shape[0]  # read frame height
    w = old_frame.shape[1]  # read frame weight
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # read fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set output video format
    OutHornSchunck = cv2.VideoWriter("ProcessedVideo-HornSchuck " +
                                     datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") +
                                     '.mp4', fourcc, fps, (w, h))  # Set output video
    frames = []
    while (1):
        ret, new_frame = cap.read()
        num_frame = num_frame + 1  # count number of frame so far
        if ret == False:
            break
        gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 15, 3, 5, 1.2, 0)  # calculate optical flow
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        ###### to output raw data as reshaped into 2 * (m * n) matrix ######
        np.savetxt(f"flow_x_frame{num_frame}.txt", flow_x, delimiter=" ")  ###### save current flow_x matrix as txt file locally
        np.savetxt(f"flow_y_frame{num_frame}.txt", flow_y, delimiter=" ")  ###### save current flow_y matrix as txt file locally
        # total motion add up, using the flow matrix
        totalmotion_count = np.sum(flow, axis=0)  # summation of every column
        totalmotion_count = np.sum(totalmotion_count, axis=0)  # summation of every row
        totalmotion_x = totalmotion_count[0] / (h * w)  # total motion along x axis
        totalmotion_y = totalmotion_count[1] / (h * w)  # total motion along y axis
        Arr_x.append(totalmotion_x)  # store this frame's total motion x component
        Arr_y.append(totalmotion_y)  # store this frame's total motion y component
        Arr_net.append(math.sqrt(totalmotion_x ** 2 + totalmotion_y ** 2))  # store this frame's net total motion
        # to draw optical flow:
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        line = []

        def function_tmp(l):
            if l[0][0] - l[1][0] > 3 or l[0][1] - l[1][1] > 3:
                line.append(l)

        list(map(function_tmp, lines))  # map optical flow line
        cv2.polylines(old_frame, line, 0, (0, 255, 255))
        frames.append(old_frame)
        OutHornSchunck.write(old_frame)
        old_frame = new_frame.copy()

    #newpath = r'C:\Desktop\result'
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)
    
    # construct csv file to output every frame's Horn Schunck Motion
    headers = ['number of frame', 'current time (s)', 'total motion mean_x',
               'total motion mean_y', 'net total motion  velocity']
    with open('Horn_Schunck_Motion.' + datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.csv', 'w',
              newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(len(Arr_x)):
            f_csv.writerow([i + 1, (i+1) / fps, Arr_x[i], Arr_y[i], Arr_net[i]])

    cap.release()
    OutHornSchunck.release()  # release processed video
    for i in frames:
        cv2.imshow("Optical flow HS", i)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    result1.set("Video with optical flow vectors successfully constructed!")
    result_tag1.set("csv file successfully constructed!")
    result2.set("Please check at")
    result_tag2.set("current folder!")

def OpticalFlowLucasKanade(route):
    result1.set("Pending")
    result_tag1.set("Pending")
    result2.set("Pending")
    result_tag2.set("Pending")
    cap = cv2.VideoCapture(route)
    if cap.isOpened() is False:
        result1.set("Video open")
        result_tag1.set("failed :(")
        result2.set("Check your route")
        result_tag2.set("please ^^")
        return
    hashvalue = hash_value_entry.get()
    if (len(hashvalue) == 0):
        hashvalue = 23
    else:
        hashvalue = float(hashvalue)
    list = []  # create an empty list to store every last second's hash change values
    ret, old_frame = cap.read()  # get first frame
    h = old_frame.shape[0]  # read frame height
    w = old_frame.shape[1]  # read frame weight
    fps =int(cap.get(cv2.CAP_PROP_FPS))  # read video fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set output video format
    OutLucasKanade = cv2.VideoWriter("ProcessedVideo-LucasKanade " +
                                     datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") +
                                     '.mp4', fourcc, fps, (w, h))  # Set output video
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)  # ShiTomasi Set parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # Set LKOF parameters
    color = np.random.randint(0, 255, (100, 3))  # Create random colors
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # transform first frame to GRAY style(returning corner points needs single channel image)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # Get corner points
    old_PIL = Image.fromarray(cv2.cvtColor(old_frame, cv2.COLOR_BGRA2RGB))
    h0 = imagehash.average_hash(old_PIL)  # calculate old frame hash value
    mask = np.zeros_like(old_frame)  # Create background to draw trajectories

    frames = []
    while(1):  # loop doesn't end until break
        ret, frame = cap.read()  # read next frame
        if ret == False:  # no frame read
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Transform to GRAY style
        new_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB))
        h1 = imagehash.average_hash(new_PIL)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # Calculate Optical Flow
        delta_y = abs(h0 - h1)
        lenlist = len(list)
        if lenlist == 0:
            hc_mean = 0.0
        else:
            if lenlist <= (fps / 4):
                hc_mean = sum(list) / lenlist
            else:
                list.pop(0)  # delete one element that is prior to the last second
                hc_mean = sum(list) / (lenlist - 1)
        list.append(delta_y)  # store hash change value just calculated
        if (delta_y >= hashvalue and delta_y >= 4.5 * hc_mean and p1[st == 1].size != 0):
            good_new = p1[st == 1]  # Select good points from new frame
            good_old = p0[st == 1]  # Select corresponding points from old frame
            # Draw Trajectories:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # coordinate on new frame
                c, d = old.ravel()  # coordinate on old frame
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # Draw Line
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)  # Draw circle
            img = cv2.add(frame, mask)  # overlap frame and mask
            cv2.imshow('frame', img)  # Show image
            OutLucasKanade.write(img)  # Write out each frame
            old_gray = frame_gray.copy()  # prepare for reading next frame
            p0 = good_new.reshape(-1,1,2)  # return a numpy array with shape = (n, 1, 2)
            h0 = h1  # set h0 to the latest hash value for next loop
        else:
            mask = np.zeros_like(old_frame)  # Create background to draw trajectories
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)  # Get new corner points
            ret, frame_next = cap.read()  # read next frame
            if ret == False:  # no frame read
                break
            frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, frame_next_gray, p0, None,
                                                   **lk_params)  # Calculate Optical Flow
            good_new = p1[st == 1]  # Select good points from new frame
            good_old = p0[st == 1]  # Select corresponding points from old frame
            # Draw Trajectories:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # coordinate on new frame
                c, d = old.ravel()  # coordinate on old frame
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # Draw Line
                frame = cv2.circle(frame_next, (a, b), 5, color[i].tolist(), -1)  # Draw circle
            img = cv2.add(frame, mask)  # overlap frame and mask
            cv2.imshow('frame', img)  # Show image
            OutLucasKanade.write(img)  # Write out each frame
            old_gray = frame_next_gray.copy()  # prepare for reading next frame
            p0 = good_new.reshape(-1, 1, 2)  # return a numpy array with shape = (n, 1, 2)
            old_PIL = Image.fromarray(cv2.cvtColor(frame_next, cv2.COLOR_BGRA2RGB))
            h0 = imagehash.average_hash(old_PIL)  # calculate old frame hash value for next loop

    cap.release()
    OutLucasKanade.release()  # release processed video
    for i in frames:
        cv2.imshow("Optical flow Lucas Kanade", i)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    result1.set("Video")
    result_tag1.set("constructed !")
    result2.set("Please check at")
    result_tag2.set("current folder!")


root = Tk()
root.title("Video Processing ToolBox")
mainframe = tk.Frame(root, padx=30, pady=30)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

route = StringVar()
route_entry = tk.Entry(mainframe, width=5, textvariable=route)
route_entry.grid(column=3, row=1, sticky=(W, E))
tk.Label(mainframe, text="Video Path").grid(column=2, row=1, sticky=W)

shot_changing = StringVar()
shot_changing_entry = tk.Entry(mainframe, width=5, textvariable=shot_changing)
shot_changing_entry.grid(column=5, row=1, sticky=(W, E))
tk.Label(mainframe, text="ShotChangeSites/s_optional eg:(1st,...,end)").grid(column=4, row=1, sticky=(W, E))

hash_value = StringVar()
hash_value_entry = tk.Entry(mainframe, width=5, textvariable=hash_value)
hash_value_entry.grid(column=7, row=1, sticky=(W, E))
tk.Label(mainframe, text="HashValue_optional").grid(column=6, row=1, sticky=(W, E))

bin_num = StringVar()
bin_num_entry = tk.Entry(mainframe, width=5, textvariable=bin_num)
bin_num_entry.grid(column=9, row=1, sticky=(W, E))
tk.Label(mainframe, text="number_optional").grid(column=8, row=1, sticky=(W, E))

# section1
tk.Label(mainframe, text="IMAGE / FRAME").grid(column=5, row=4, sticky=W)
tk.Button(mainframe, text="Black-White Ratio", command=lambda: BlacktoWhiteRatio(route_entry.get())).grid(column=2,
                                                                                                          row=5,
                                                                                                          sticky=W)
tk.Button(mainframe, text="Luminosity", command=lambda: Luminosity(route_entry.get())).grid(column=3,
                                                                                            row=5,
                                                                                            sticky=W)
tk.Button(mainframe, text="Saturation", command=lambda: Saturation(route_entry.get())).grid(column=4,
                                                                                            row=5,
                                                                                            sticky=W)
tk.Button(mainframe, text="Chromatic variety", command=lambda: ChromaticVariety(route_entry.get())).grid(column=5,
                                                                                                         row=5,
                                                                                                         sticky=W)
tk.Button(mainframe, text="Entropy of luminosity", command=lambda: EntropyofLuminosity(route_entry.get())).grid(
    column=6,
    row=5,
    sticky=W)
tk.Button(mainframe, text="Contrast", command=lambda: Contrast(route_entry.get())).grid(column=7,
                                                                                        row=5,
                                                                                        sticky=W)
tk.Button(mainframe, text="ImageFeaturesALL", command=lambda: LowLevelFeaturesALL(route_entry.get())).grid(
                                                                                        column=8,
                                                                                        row=5,
                                                                                        sticky=W)
tk.Button(mainframe, text="Black-White Ratio_EF", command=lambda: BlacktoWhiteRatioEF(route_entry.get())).grid(
    column=2,
    row=6,
    sticky=W)
tk.Button(mainframe, text="Luminosity_EF", command=lambda: LuminosityEF(route_entry.get())).grid(column=3,
                                                                                                 row=6,
                                                                                                 sticky=W)
tk.Button(mainframe, text="Saturation_EF", command=lambda: SaturationEF(route_entry.get())).grid(column=4,
                                                                                                 row=6,
                                                                                                 sticky=W)
tk.Button(mainframe, text="Chromatic variety_EF", command=lambda: ChromaticVarietyEF(route_entry.get())).grid(column=5,
                                                                                                              row=6,
                                                                                                              sticky=W)
tk.Button(mainframe, text="Entropy of luminosity_EF", command=lambda: EntropyofLuminosityEF(route_entry.get())).grid(
    column=6,
    row=6,
    sticky=W)
tk.Button(mainframe, text="Contrast_EF", command=lambda: ContrastEF(route_entry.get())).grid(column=7,
                                                                                             row=6,
                                                                                             sticky=W)
tk.Button(mainframe, text="ImageFeaturesALL_EF", command=lambda: LowLevelFeaturesALLEF(route_entry.get())).grid(
                                                                                        column=8,
                                                                                        row=6,
                                                                                        sticky=W)

# section 2
tk.Label(mainframe, text="PACE").grid(column=5, row=7, sticky=W)
tk.Button(mainframe, text="AverageShotLength", command=lambda: AverageShotLength(route_entry.get())).grid(column=2,
                                                                                                          row=8,
                                                                                                          sticky=W)
tk.Button(mainframe, text="ShotLengthVariance", command=lambda: VarianceOfShotLength(route_entry.get())).grid(column=3,
                                                                                                              row=8,
                                                                                                              sticky=W)
tk.Button(mainframe, text="MedianShotLength", command=lambda: MedianShotLength(route_entry.get())).grid(column=4,
                                                                                                        row=8,
                                                                                                        sticky=W)

tk.Button(mainframe, text="ShotLengthALL", command=lambda: shotlengthALL(route_entry.get())).grid(column=5,
                                                                                                  row=8,
                                                                                                  sticky=W)

tk.Button(mainframe, text="video's length", command=lambda: runtime(route_entry.get())).grid(column=6,
                                                                                             row=8,
                                                                                             sticky=W)
tk.Button(mainframe, text="FadeRate", command=lambda: FadeRate(route_entry.get())).grid(column=7,
                                                                                        row=8,
                                                                                        sticky=W)
tk.Button(mainframe, text="DissolveRate", command=lambda: DissolveRate(route_entry.get())).grid(column=8,
                                                                                                row=8,
                                                                                                sticky=W)

# section 3
tk.Label(mainframe, text="MOTION").grid(column=5, row=9, sticky=W)
tk.Button(mainframe, text="CameraMotionIntensity", command=lambda: CameraMotionIntensity(route_entry.get())).grid(
    column=2,
    row=10,
    sticky=W)
tk.Button(mainframe, text="CameraMotionIntensityVariance",
          command=lambda: CameraMotionIntensityVariance(route_entry.get())).grid(column=3,
                                                                                 row=10,
                                                                                 sticky=W)
tk.Button(mainframe, text="CameraMotionComplexity", command=lambda: CameraMotionComplexity(route_entry.get())).grid(
    column=4,
    row=10,
    sticky=W)

tk.Button(mainframe, text="CameraMotionComplexityVariance",
          command=lambda: CameraMotionComplexityVariance(route_entry.get())).grid(column=5,
                                                                                  row=10,
                                                                                  sticky=W)

tk.Button(mainframe, text="CameraMotionALL", command=lambda: CameraMotionAll(route_entry.get())).grid(column=6,
                                                                                                      row=10,
                                                                                                      sticky=W)


tk.Button(mainframe, text="InternalMotionIntensity", command=lambda: InternalMotionIntensity(route_entry.get())).grid(
    column=2,
    row=11,
    sticky=W)
tk.Button(mainframe, text="InternalMotionIntensityVariance",
          command=lambda: InternalMotionIntensityVariance(route_entry.get())).grid(column=3,
                                                                                   row=11,
                                                                                   sticky=W)
tk.Button(mainframe, text="InternalMotionComplexity", command=lambda: InternalMotionComplexity(route_entry.get())).grid(
    column=4,
    row=11,
    sticky=W)
tk.Button(mainframe, text="InternalMotionComplexityVariance",
          command=lambda: InternalMotionComplexityVariance(route_entry.get())).grid(column=5,
                                                                                    row=11,
                                                                                    sticky=W)
tk.Button(mainframe, text="InternalMotionALL", command=lambda: InternalMotionAll(route_entry.get())).grid(column=6,
                                                                                                          row=11,
                                                                                                          sticky=W)

# section3 - Optical Flow
tk.Label(mainframe, text="Optical Flow:").grid(column=2, row=12, sticky=W)
tk.Button(mainframe, text="OpticalFlowHornSchunck", command=lambda: OpticalFlowHornSchunck(route_entry.get())).grid(
    column=3,
    row=12,
    sticky=W)
tk.Button(mainframe, text="OpticalFlowLucasKanade", command=lambda: OpticalFlowLucasKanade(route_entry.get())).grid(
    column=4,
    row=12,
    sticky=W)

result1 = StringVar()
tk.Label(mainframe, textvariable=result1).grid(column=3, row=2, sticky=W)
result2 = StringVar()
tk.Label(mainframe, textvariable=result2).grid(column=3, row=3, sticky=W)
result_tag1 = StringVar()
tk.Label(mainframe, textvariable=result_tag1).grid(column=2, row=2, sticky=W)
result_tag2 = StringVar()
tk.Label(mainframe, textvariable=result_tag2).grid(column=2, row=3, sticky=W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=8, pady=8)

route_entry.focus()
root.bind("<Return>", BlacktoWhiteRatio)
root.mainloop()
