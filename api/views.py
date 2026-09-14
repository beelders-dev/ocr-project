import os
import tempfile

from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ReceiptUploadSerializer
from src.inference import predict_words


class ReceiptProcessView(APIView):
    def post(self, request):
        serializer = ReceiptUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image = serializer.validated_data["image"]
        suffix = os.path.splitext(image.name)[1]

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        try:
            fields = predict_words(temp_path)

            return Response(
                {
                    "success": True,
                    "data": fields,
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()

            return Response(
                {
                    "success": False,
                    "error": str(e),
                },
                status=500,
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
