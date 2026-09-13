from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import ReceiptUploadSerializer


class ReceiptProcessView(APIView):
    def post(self, request):
        serializer = ReceiptUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image = serializer.validated_data["image"]

        return Response(
            {
                "message": "Receipt received successfully.",
                "filename": image.name,
            }
        )
