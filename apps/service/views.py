import base64
import io

from PIL import Image
from rest_framework import status
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from apps.service.emotion_recognition import emotion_recognition


class UploadImageViewSet(GenericAPIView):

    def post(self, request, *args, **kwargs):
        img = request.data['img']
        arr = bytes(img, encoding='utf-8')
        decoded_bytes = base64.decodebytes(arr)
        image_stream = io.BytesIO(decoded_bytes)
        image = Image.open(image_stream)

        emotion = emotion_recognition(image)

        return Response(
            {
                'message': 'The emotion is: ' + emotion,
                'emotion': emotion
            },
            status=status.HTTP_200_OK
        )
