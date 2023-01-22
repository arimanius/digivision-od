# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from od.api.v1 import object_detector_pb2 as od_dot_api_dot_v1_dot_object__detector__pb2


class ObjectDetectorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Detect = channel.unary_unary(
                '/object_detector.ObjectDetector/Detect',
                request_serializer=od_dot_api_dot_v1_dot_object__detector__pb2.Image.SerializeToString,
                response_deserializer=od_dot_api_dot_v1_dot_object__detector__pb2.BoundingBox.FromString,
                )


class ObjectDetectorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectDetectorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Detect': grpc.unary_unary_rpc_method_handler(
                    servicer.Detect,
                    request_deserializer=od_dot_api_dot_v1_dot_object__detector__pb2.Image.FromString,
                    response_serializer=od_dot_api_dot_v1_dot_object__detector__pb2.BoundingBox.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'object_detector.ObjectDetector', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ObjectDetector(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Detect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/object_detector.ObjectDetector/Detect',
            od_dot_api_dot_v1_dot_object__detector__pb2.Image.SerializeToString,
            od_dot_api_dot_v1_dot_object__detector__pb2.BoundingBox.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
