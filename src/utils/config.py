import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:

    # 공공데이터포털 API KEY
    PUBLIC_DATA_API_KEY = os.getenv("PUBLIC_DATA_API_KEY")

    # 성분 API URL
    COSMETIC_INGREDIENT_API_URL = os.getenv(
        "COSMETIC_INGREDIENT_API_URL"
    )


config = Config()