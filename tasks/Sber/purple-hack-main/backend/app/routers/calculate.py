from fastapi import APIRouter, Query, UploadFile, File
from ..utils.calculate import process_json
import json

router = APIRouter()


@router.post("/calculate")
async def calculate_tasks(
        duration: float = Query(..., description="Duration parameter"),
        price: float = Query(..., description="Price parameter"),
        resource: float = Query(..., description="Resource parameter"),
        json_file: UploadFile = File(..., description="JSON file")
):
    content = await json_file.read()

    return process_json(json.loads(content.decode("utf-8")), duration, price, resource)
