from pydantic import BaseModel,Field
from typing import Literal

class YesNo(BaseModel):
  is_topic_query:Literal["YES","NO"]=Field(...,description="respod only with the YES or NO in the json format")  #... indicates required


