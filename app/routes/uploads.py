import uuid
from http.client import HTTPException

from fastapi import APIRouter, Depends

from app.dependencies.s3 import get_s3_client, S3_BUCKET

router = APIRouter()

@router.get("/presignedurl", tags=["uploads"])
async def get_presigned_url(
        s3=Depends(get_s3_client)
):
    unique_id = str(uuid.uuid4())
    knowledge_base_data_key = f"{unique_id}/kb_data/kb.pdf"
    ground_truth_data_key = f"{unique_id}/gt_data/gt.json"

    kb_data_path = f"s3://{S3_BUCKET}/{knowledge_base_data_key}"
    gt_data_path = f"s3://{S3_BUCKET}/{ground_truth_data_key}"

    try:
        kb_data_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": S3_BUCKET, "Key": knowledge_base_data_key},
            ExpiresIn=7200,
        )

        gt_data_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": S3_BUCKET, "Key": ground_truth_data_key},
            ExpiresIn=7200,
        )

        return {
            "kb_data": {
                "path": kb_data_path,
                "presignedurl": kb_data_url
            },
            "gt_data": {
                "path": gt_data_path,
                "presignedurl": gt_data_url
            },
            "uuid": unique_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pre-signed URL: {str(e)}")
