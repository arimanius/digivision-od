# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: od/api/v1/object_detector.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fod/api/v1/object_detector.proto\x12\x0fobject_detector\"\x16\n\x05Image\x12\r\n\x05image\x18\x01 \x01(\x0c\" \n\x08Position\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\"L\n\x0e\x44\x65tectResponse\x12\x10\n\x08is_found\x18\x01 \x01(\x08\x12(\n\x02\x62\x62\x18\x02 \x01(\x0b\x32\x1c.object_detector.BoundingBox\"k\n\x0b\x42oundingBox\x12+\n\x08top_left\x18\x01 \x01(\x0b\x32\x19.object_detector.Position\x12/\n\x0c\x62ottom_right\x18\x02 \x01(\x0b\x32\x19.object_detector.Position2R\n\x0eObjectDetector\x12@\n\x06\x44\x65tect\x12\x16.object_detector.Image\x1a\x1c.object_detector.BoundingBox\"\x00\x62\x06proto3')



_IMAGE = DESCRIPTOR.message_types_by_name['Image']
_POSITION = DESCRIPTOR.message_types_by_name['Position']
_DETECTRESPONSE = DESCRIPTOR.message_types_by_name['DetectResponse']
_BOUNDINGBOX = DESCRIPTOR.message_types_by_name['BoundingBox']
Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'od.api.v1.object_detector_pb2'
  # @@protoc_insertion_point(class_scope:object_detector.Image)
  })
_sym_db.RegisterMessage(Image)

Position = _reflection.GeneratedProtocolMessageType('Position', (_message.Message,), {
  'DESCRIPTOR' : _POSITION,
  '__module__' : 'od.api.v1.object_detector_pb2'
  # @@protoc_insertion_point(class_scope:object_detector.Position)
  })
_sym_db.RegisterMessage(Position)

DetectResponse = _reflection.GeneratedProtocolMessageType('DetectResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTRESPONSE,
  '__module__' : 'od.api.v1.object_detector_pb2'
  # @@protoc_insertion_point(class_scope:object_detector.DetectResponse)
  })
_sym_db.RegisterMessage(DetectResponse)

BoundingBox = _reflection.GeneratedProtocolMessageType('BoundingBox', (_message.Message,), {
  'DESCRIPTOR' : _BOUNDINGBOX,
  '__module__' : 'od.api.v1.object_detector_pb2'
  # @@protoc_insertion_point(class_scope:object_detector.BoundingBox)
  })
_sym_db.RegisterMessage(BoundingBox)

_OBJECTDETECTOR = DESCRIPTOR.services_by_name['ObjectDetector']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _IMAGE._serialized_start=52
  _IMAGE._serialized_end=74
  _POSITION._serialized_start=76
  _POSITION._serialized_end=108
  _DETECTRESPONSE._serialized_start=110
  _DETECTRESPONSE._serialized_end=186
  _BOUNDINGBOX._serialized_start=188
  _BOUNDINGBOX._serialized_end=295
  _OBJECTDETECTOR._serialized_start=297
  _OBJECTDETECTOR._serialized_end=379
# @@protoc_insertion_point(module_scope)
