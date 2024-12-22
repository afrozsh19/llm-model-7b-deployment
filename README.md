# LLM Model Deployment

The repository provides a scalable and fault tolerant way to deploy a Large Language Model (LLM) using helm charts.
The solution integrates optional Prometheus monitoring, supports PVC-based model storage, and includes advanced scaling features.

## Motivation
The idea was to build a robust and efficient model-serving infrastructure. This comes with a small amount of challenges and complexities.

1. Model Size and Persistence:
    - LLMs often come with high memory utilisation, making ephemeral storage nearly impractical for each pod.
    - Using Persistent Volume Claim (PVC) or an external storage mechanism allows shared storage that ensures models are securely loaded once and reused across multiple pods. This is also essential since every model download from source like HuggingFace would mean an extract of new model artefact. Therefore, a pre-downloaded model ensures models are safely loaded from a controlled environment.
2. Multiple Devices Across Platforms:
    - For every working environment the model files need to be loaded with its distinct configuration.
    - The FastAPI application service, identifies the current device in use and utilises the same with transformers library for model sharding and off loading.
        - CPU for low-end environments
        - GPU for accelerated inferences
        - MPS for Apple Silicon devices
3. Fault Tolerance:
    - Liveness and readiness probes are configured to call the health check endpoint in the FastAPI application, that makes a model inference call in the background to ensure the service is up and ready to serve. 

## Features
### 1. Automated Docker Image Generation
- Github workflow is leverage to trigger build and push of docker images on every pull request into main branch of the repository.
- The Github workflow calls a reusable docker workflow in a my private repository [afrozsh19/githubWorkflows](https://github.com/afrozsh19/githubWorkflows/tree/main/.github/workflows) and leverages the workflow step to safely build and push the docker image to container repository.
- The workflow can be accessed [here](https://github.com/afrozsh19/llm-model-7b-deployment/blob/main/.github/workflows/build-docker-images.yaml) in the repository.
    ![Automated Docker Workflow](img/automated-docker-workflow.png)

### 2. Model Management:
- Models are pre-downloaded into a shared PVC using an init container in the [deployment configuration](https://github.com/afrozsh19/llm-model-7b-deployment/blob/main/model-api-app/templates/deployment.yaml#L38)
- Model Download script can adapt to any hugging face model downloads. Below environment variables are required for the script to be functional:
    - `MODEL_NAME`: `.Values.model.name` : Name of the model in hugging face. For example: `mistralai/Mistral-7B-v0.1`
    - `MODEL_DIR`: `.Values.model.path` : Path to the directory where the model will be downloaded.
    - `HF_TOKEN`: `.Values.secret.hfToken` : HuggingFace acces token.
    - `LOG_LEVEL`: `.Values.logger.level` : Logging Level to see the container logs.

#### 2.1 Low-Precision Inference (Device-Agnostic Execution)
- Used `bitsandbytes` for 8-bit model loading to reduce the memory footprint. This is a configurable option in the helm chart `.Values.model.bitAndBytes.enabled`, which would generate a quantization config in the model loading function.
- Implemented logic to offload parts of the model to CPU if GPU memory is insufficient using `device_map` which is a part of accelerate configuration.
    ```
    model = AutoModelForCausalLM.from_pretrained(
            CONFIG.get("model_path"),
            device_map="auto",
            quantization_config = BitsAndBytesConfig(load_in_8bit=True),
            local_files_only=True
        )
    ```

#### 2.2 Handling Input Tensors (Device-Agnostic Execution)
- During model inferencing, it is necessary to move the inputs tensors to the same device in case of mps device. This is handled by identifying the device at run time and assigning it to inference logic.
```
def get_device_in_use():
    """
    Returns the current devide that the code is running.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### 3. Autoscaling
Horizontal Pod Autoscaler adjusts the replicas based on the resource utilisation of the deployment. This is a configurable option and can be enabled or disabled using helm chart values.
```
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80
```

### Platform Agnoticism:
Resource level configurations along with the node selector, affinities and tolerations are configurable in the deployment and can be updated using [values file](https://github.com/afrozsh19/llm-model-7b-deployment/blob/main/model-api-app/values.yaml#L126).

For example the resources, node selector and tolerations could be configured to use GPU as below:
```
resources:
  requests:
    memory: "18Gi"
    cpu: "2"
    nvidia.com/gpu: '1'
  limits:
    memory: "36Gi"
    cpu: "2"
    nvidia.com/gpu: '1'
nodeSelector:
    nodepool: "nd40rsv2" # which for example is a V100 instance in the Azure Cloud
tolerations:
  - effect: NoSchedule
    key: sku
    operator: Equal
    value: gpu
  - effect: NoSchedule
    key: nvidia.com/gpu
    operator: Exists
``` 

### Environment Specific Configuration
You could utilise additional values files to override the environment specific configurations as demonstrated in the [values-prod.yaml](https://github.com/afrozsh19/llm-model-7b-deployment/blob/main/model-api-app/overlays/values-prod.yaml)

