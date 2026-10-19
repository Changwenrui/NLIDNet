import socket
import sys
import os
import time
import logging
import argparse
from pathlib import Path
import yaml

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


queryInterval = 0.01
startx, starty, widthx, hieghty = 410, 0, 500, 600
bias_diff_on = 20
bias_diff_off = 20


rgbCameraExposureTime = 25000
rgbCameraMaxWidth = 1280
rgbCameraMaxHeight = 1024
rgbCameraWidth = 600
rgbCameraHeight = 600
rgbCameraXOffset = 264
rgbCameraYOffset = 0


waitTimeAfterEndRotateSignal = 2
inf = 100000


def recv(sock, buffer_size=1024):
    data = sock.recv(buffer_size)
    return data.decode().strip()


def sendStartSignal():
    ip = robot_ip
    s = socket.socket()
    s.connect((ip, 29999))

    data = recv(s)
    assert data == "Connected: Universal Robots Dashboard Server"

    s.sendall(b"play\n")
    data = recv(s)
    assert data == "Starting program"

    s.close()
    logging.info("end sendStartSignal")


class EventCamera:
    def __init__(self, filePath):
        logging.info("start EventCamera __init__")
        from metavision_core.event_io.raw_reader import initiate_device
        import metavision_hal

        device = initiate_device("")

        device.get_i_ll_biases().set("bias_diff_on", bias_diff_on)
        device.get_i_ll_biases().set("bias_diff_off", bias_diff_off)

        windowroi = metavision_hal.I_ROI.Window(startx, starty, widthx, hieghty)
        device.get_i_roi().enable(1)
        device.get_i_roi().set_window(windowroi)

        if device.get_i_events_stream():
            device.get_i_events_stream().log_raw_data(filePath)
        self.device = device
        logging.info("end EventCamera __init__")

    def startCamera(self):
        logging.info("start EventCamera startCamera")
        device = self.device
        delta_t = queryInterval * 1e6
        from metavision_core.event_io import EventsIterator

        mv_iterator = EventsIterator.from_device(device=device, delta_t=delta_t)
        self.mv_iterator = mv_iterator
        logging.info("end EventCamera startCamera")
        return mv_iterator

    def endCamera(self):
        logging.info("start EventCamera endCamera")

        self.device.get_i_events_stream().stop_log_raw_data()
        logging.info("end EventCamera endCamera")


class RGBCamera:
    def __init__(self, folderPath):
        logging.info("start RGBCamera __init__")
        camera_index = "1"

        from hardware.acquisition.rgb_camera import CameraController

        controller = CameraController(dir=folderPath)
        self.controller = controller

        try:
            controller.enumerate_devices()
            result = controller.start_camera(camera_index, rgbCameraExposureTime)
            if not result:
                raise Exception("Failed to start camera")

        except Exception as e:
            controller.stop_camera()
            raise e

        logging.info("end RGBCamera __init__")

    def startCamera(self):
        logging.info("start RGBCamera startCamera")
        self.controller.startGrabbing()
        logging.info("end RGBCamera startCamera")

    def endCamera(self):
        logging.info("start RGBCamera endCamera")
        self.controller.stop_camera()
        logging.info("end RGBCamera endCamera")

    def saveAllImage(self):
        maxSize = rgbCameraMaxWidth, rgbCameraMaxHeight
        roi = rgbCameraXOffset, rgbCameraYOffset, rgbCameraWidth, rgbCameraHeight
        self.controller.saveAllImage(maxSize, roi)


def receiveData(connect, blocking):
    connect.setblocking(blocking)
    try:
        data = connect.recv(1024)
    except BlockingIOError:
        return ""

    dataDecode = data.decode().strip()
    logging.info(f"dataDecode: {dataDecode}")
    return dataDecode


def configure(cfg):
    global robot_ip, listen_port
    global queryInterval, startx, starty, widthx, hieghty, bias_diff_on, bias_diff_off
    global rgbCameraExposureTime, rgbCameraMaxWidth, rgbCameraMaxHeight
    global rgbCameraWidth, rgbCameraHeight, rgbCameraXOffset, rgbCameraYOffset
    robot_ip, listen_port = cfg["robot_ip"], cfg["listen_port"]
    queryInterval = cfg["event_query_interval_s"]
    startx, starty, widthx, hieghty = cfg["event_roi_xywh"]
    bias_diff_on, bias_diff_off = cfg["bias_diff_on"], cfg["bias_diff_off"]
    rgbCameraExposureTime = cfg["rgb_exposure_us"]
    rgbCameraMaxWidth, rgbCameraMaxHeight = cfg["rgb_sensor_size"]
    rgbCameraXOffset, rgbCameraYOffset, rgbCameraWidth, rgbCameraHeight = cfg[
        "rgb_roi_xywh"
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Record one synchronized pouring trial"
    )
    parser.add_argument("event_file")
    parser.add_argument("rgb_folder")
    parser.add_argument("--config", default="hardware/config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    configure(cfg)
    eventFilePath, rgbFolderPath = args.event_file, args.rgb_folder
    Path(eventFilePath).parent.mkdir(parents=True, exist_ok=True)
    if Path(eventFilePath).exists():
        raise FileExistsError(eventFilePath)
    logging.info(f"eventFilePath: {eventFilePath}   rgbFolderPath: {rgbFolderPath}")

    if os.path.exists(rgbFolderPath) and os.listdir(rgbFolderPath):
        logging.error(f"Folder {rgbFolderPath} already exists")
        raise FileExistsError(f"Folder {rgbFolderPath} already exists")
    os.makedirs(rgbFolderPath, exist_ok=True)

    HOST = ""
    PORT = listen_port

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    sendStartSignal()

    logging.info("waiting for connection")
    conn, addr = s.accept()
    logging.info(f"connection established with {addr}")

    data = receiveData(conn, True)
    assert data == "start program"

    eventCamera = EventCamera(eventFilePath)
    rgbCamera = RGBCamera(rgbFolderPath)

    startReceiveTime = time.time()
    data = receiveData(conn, True)
    endReceiveTime = time.time()

    if endReceiveTime - startReceiveTime < 1:
        logging.error("not receive start rotate signal in time")
        exit(55)
    assert data == "start rotate"

    rgbCamera.startCamera()
    mv_iterator = eventCamera.startCamera()

    stopIndex = inf

    for i, evs in enumerate(mv_iterator):
        if i >= stopIndex + waitTimeAfterEndRotateSignal / queryInterval:
            eventCamera.endCamera()
            rgbCamera.endCamera()
            break
        if i % 100 == 0:
            logging.info(f"recording {i*queryInterval} s")

        data = receiveData(conn, False)
        assert data == "" or data == "end rotate"
        if data == "end rotate":
            stopIndex = i

    time.sleep(2)
    rgbCamera.saveAllImage()

    data = receiveData(conn, True)
    assert data == "end program"

    conn.close()
    logging.info("connection closed")


if __name__ == "__main__":
    main()
