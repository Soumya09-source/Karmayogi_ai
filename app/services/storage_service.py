from minio import Minio

from app.core.config import settings


minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False,
)


def ensure_bucket_exists():
    if not minio_client.bucket_exists(settings.MINIO_BUCKET_NAME):
        minio_client.make_bucket(settings.MINIO_BUCKET_NAME)


def upload_file(file_data, object_name: str, content_type: str):
    ensure_bucket_exists()

    minio_client.put_object(
        settings.MINIO_BUCKET_NAME,
        object_name,
        file_data,
        length=-1,
        part_size=10 * 1024 * 1024,
        content_type=content_type,
    )

    return object_name