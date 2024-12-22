import torch

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from api_schema import InferenceRequest, InferenceResponse, ErrorResponse
from utils import ( 
    validation_error_response,
    python_error_response,
    get_device_in_use,
    CONFIG )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initiliase the model and tokenizer using bitandbytes for low precision inference.
    """
    global tokenizer, model
    # logger.info("Loading tokenizer into the context")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.get("model_path"), 
        local_files_only=True
    )

    if CONFIG.get("use_bits_and_bytes"):
        # logger.info("Loading model into the context")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG.get("model_path"),
            device_map="auto",
            quantization_config = BitsAndBytesConfig(load_in_8bit=True),
            local_files_only=True
        )
    else:
        # logger.info("Loading model into the context")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG.get("model_path"),
            device_map="auto",
            local_files_only=True
        )
    model.eval()
    yield
    # logger.info("Model loaded successfully")


async def predict(request: InferenceRequest):
    """
    Infer from the model and return response
    """
    prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device_in_use())
    outputs = model.generate(inputs["input_ids"], max_length=100) # TODO: max_length can be accepted from users input
    num_output_tokens = outputs.shape[1]
    # convert a list of lists of token-ids into a list of strings
    results = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
    return results[0]


app = FastAPI(
    title = f"{CONFIG.get("model_name")} API Server",
    description = f"API for serving model {CONFIG.get("model_name")} using ray and fastAPI",
    version = "1.0.0",
    lifespan=lifespan
)

app.add_exception_handler(RequestValidationError, validation_error_response)
app.add_exception_handler(Exception, python_error_response)


@app.get("/healthz", response_model=InferenceResponse)
async def health_check():
    """
    Endpoint for service health checks
    """
    try:
        test_input = InferenceRequest(
            messages=[{"role": "system", "content": "What is 1+2"}],
            params={"max_length": 10, "temperature": 0.1}
        )
        print(test_input)
        response = await predict(test_input)
        print(response)
        if response:
            return {
                "status_code": 200, 
                "response": "Healthy!! Model is ready to serve."}
        else: 
            raise HTTPException(status_code=503, detail="Model is not ready to serve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    

@app.post("api/v1/predict", response_model=InferenceResponse, responses={500: {"model": ErrorResponse}})
async def generate(request: Request, body: InferenceRequest):
    """
    Function to handle model request
    """
    response = await predict(body)
    return {
        "status_code": 200,
        "response": response
    }


@app.get("/")
def landing_page():
    """
    Landing page for the fastApi application
    """
    return {
        "Application Name": f"{CONFIG.get("model_name")} Model Server",
        "Torch Version": torch.__version__,
        "Device in use": get_device_in_use(),
        "Model Details": model.hf_device_map
    }
