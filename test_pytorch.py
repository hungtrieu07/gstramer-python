import gi
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst
import numpy as np
import torch
from PIL import Image
import cv2

Gst.init(None)

src = Gst.ElementFactory.make('filesrc', 'source')
src.set_property('location', 'video/test.mp4')
src.set_property('num-buffers', 20000)
src.set_property('blocksize', 4096)
decodebin = Gst.ElementFactory.make('decodebin', 'decoder')
videoconvert = Gst.ElementFactory.make('videoconvert', 'converter')
queue = Gst.ElementFactory.make('queue', 'queue')
autovideosink = Gst.ElementFactory.make('autovideosink', 's')

pipeline = Gst.Pipeline.new('new-pipeline')

pipeline.add(src)
pipeline.add(decodebin)
pipeline.add(videoconvert)
pipeline.add(queue)
pipeline.add(autovideosink)

def on_pad_added(element, pad):
    sinkpad = videoconvert.get_static_pad('sink')
    pad.link(sinkpad)
    decodebin.link(queue)
    queue.link(autovideosink)
    sinkpad = autovideosink.get_static_pad('sink')
    videoconvert.link(queue)
    
src.link(decodebin)
decodebin.connect('pad-added', on_pad_added)

device = torch.device("cpu")
print(device)
detector = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt').eval().to(device)

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
    with torch.no_grad():
        detections = detector(image_tensor)
        print(detections)
        objects = (detections.xyxy[0]).tolist()
        img_np = np.array(image_tensor)
        for i in range(len(objects)):
            x_min, y_min, x_max, y_max = int(objects[i][0]), int(objects[i][1]), int(objects[i][2]), int(objects[i][3])
            cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
        # Save the first frame as an image
        Image.fromarray(img_np[:,:,:3]).save("output.jpg")
        # Remove the probe from the pad to stop processing any more frames
        # pad.remove_probe(info.id)
    return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')
    channels = 3
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            image_array = np.frombuffer(map_info.data ,dtype=np.uint8).reshape((640, 1637, 3)).copy()
            image_array.resize((height, width, channels))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, (1920, 1080))
            return Image.fromarray(image_array[:,:,:3]) # RGBA -> RGB
        finally:
            buf.unmap(map_info)

pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

while True:
    msg = pipeline.get_bus().timed_pop_filtered(
        Gst.SECOND,
        Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    if msg:
        text = msg.get_structure().to_string() if msg.get_structure() else ''
        msg_type = Gst.message_type_get_name(msg.type)
        print(f'{msg.src.name}: [{msg_type}] {text}')
        break

pipeline.set_state(Gst.State.NULL)