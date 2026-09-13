from django.urls import path

from .views import ReceiptProcessView

urlpatterns = [
    path(
        "receipts/process/",
        ReceiptProcessView.as_view(),
        name="receipt-process",
    ),
]
