from concurrent import futures
from threading import Event

import grpc
import signal

from od.config.config import Config
from od.detector import OFADetector
from od.util import getLogger
from od.api.v1 import object_detector_pb2_grpc as pb2_grpc
from od.servicer.v1.object_detector import ObjectDetectorService

logger = getLogger(__name__)


class GracefulKiller:
    kill_now = False

    def __init__(self):
        self.__event = Event()
        signal.signal(signal.SIGINT, self.__exit_gracefully)
        signal.signal(signal.SIGTERM, self.__exit_gracefully)

    def __exit_gracefully(self, signum, frame):
        self.__event.set()

    def wait(self):
        self.__event.wait()


def serve(config: Config):
    logger.info('starting server')

    killer = GracefulKiller()

    logger.info(f'loading the model: {config.ofa.model}')
    OFADetector.load(config.ofa.model)
    logger.info('model loaded')

    detector = OFADetector(config.ofa.model, config.ofa.cuda, config.ofa.bpe_dir, config.ofa.instruction)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.grpc.num_workers),
                         options=[
                             ('grpc.max_receive_message_length', 1024 ** 3),
                             ('grpc.max_send_message_length', 1024 ** 3),
                         ])
    pb2_grpc.add_ObjectDetectorServicer_to_server(ObjectDetectorService(detector), server)

    listen_addr = f'[::]:{config.grpc.listen_port}'
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f'started server on {listen_addr}')

    killer.wait()
    logger.info('stopping server')
    server.stop(0)
