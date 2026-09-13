from rest_framework import serializers


class ReceiptUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
