import io

import PIL
from PIL import Image as pi
import grpc

from od.api.v1.object_detector_pb2_grpc import ObjectDetectorServicer
from od.api.v1.object_detector_pb2 import Image, BoundingBox, Position
from od.detector import OFADetector


class ObjectDetectorService(ObjectDetectorServicer):

    def __init__(self, detector: OFADetector):
        self.__detector = detector

    def Detect(self, request: Image, context: grpc.ServicerContext) -> BoundingBox:
        try:
            image = pi.open(io.BytesIO(request.image))
            bb = self.__detector.detect(image)
            return BoundingBox(
                top_left=Position(x=max(0, int(bb[0])), y=max(0, int(bb[1]))),
                bottom_right=Position(x=min(image.width, int(bb[2])), y=min(image.height, int(bb[3]))),
            )
        except PIL.UnidentifiedImageError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f'invalid image type: {e}')
        except Exception as e:
            context.abort(grpc.StatusCode.UNKNOWN, f'unknown error: {e}')
