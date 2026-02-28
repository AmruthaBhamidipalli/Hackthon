from pydantic import BaseModel

class BatchInput(BaseModel):
    Granulation_Time: float
    Binder_Amount: float
    Drying_Temp: float
    Drying_Time: float
    Compression_Force: float
    Machine_Speed: float
    Lubricant_Conc: float
    Moisture_Content: float