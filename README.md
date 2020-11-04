# Pytorch RL on Google AI example

This repository contains a quick example on how to push a `PyTorch` based, deep q-learning
(DQL) job to the Google ai platform while viewing the results using `tensorboard`. It was based on the
[Google container documentation](https://cloud.google.com/ai-platform/training/docs/custom-containers-training) and uses the code of [the DQL Pong example of @colinskow's move37 class](https://github.com/colinskow/move37). This repository is not meant to be an in-depth review of either Reinforcement learning (RL), docker or
Google cloud but mainly serves as a template for performing PyTorch based RL in the cloud. For more information on these topics, see the following resources:

- Check out the videos of [@colinskow for an excellent overview of RL](https://www.youtube.com/watch?v=14BfO5lMiuk&list=PLWzQK00nc192L7UMJyTmLXaHa3KcO0wBT).
- Check out the video of [@kodekloud for an introduction into docker](https://www.youtube.com/watch?v=zJ6WbK9zFpI&t=3488s).
- For help with setting up and using the Google cloud platform, please checkout to the [Google documentation](https://cloud.Google.com/ai-platform/docs/).

## Requirements

- A [Google cloud account](https://cloud.Google.com/free).
- A [Google project for which billing is enabled](https://cloud.Google.com/resource-manager/docs/creating-managing-projects).
- The [Google SDK](https://cloud.Google.com/sdk/docs).
- [Docker](https://docs.docker.com/engine/install/ubuntu/).
- [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker#quickstart) (OPTIONAL).
- [Available global GPU quota](https://cloud.Google.com/compute/quotas) (OPTIONAL).

## How does this work

In order to train RL algorithms on the Google ai platform we need the following components:

- A python training script.
- A python requirements file.
- A docker file.
- [The Tensorboard package](https://pypi.org/project/tensorboard/).
- Access to the [Google container registry](https://cloud.Google.com/container-registry/docs/quickstart).
- A [Google cloud bucket](https://cloud.Google.com/storage/docs/json_api/v1/buckets).

### The python training script

For the training script, I used the `dqn_basic` script of [@colinskow's move37 class](https://github.com/colinskow/move37). Two small modifications were made to this script to use it with the Google AI platform. First, I added the `model_dir` argument to the script to allow us to specify the Google cloud bucket location where we want to store the model and the TensorFlow logs:

```python
parser.add_argument(
    "--model-dir", default=".", help="The directory to store the model"
)
```

Following, I used this argument to set the `log_dir` of the tensorboard `SummaryWriter` object:

```python
writer = SummaryWriter(
    comment="-" + args.env,
    log_dir=os.path.join(args.model_dir if args.model_dir else ".", model_dir_name),
)
```

Lastly, I used the `gsutil` module to write the trained model to the Google cloud bucket that is specified in the `model_dir` argument:

```python
  retval = subprocess.check_call(
      [
          "gsutil",
          "cp",
          tmp_model_file,
          os.path.join(args.model_dir, tmp_model_file),
      ],
      stdin=sys.stdout,
      stderr=sys.stdout,
  )
  if retval > 0:
      raise Exception(
          "Could not save model as. Supplied Google cloud "
          "bucket does not exists! Shutting down training."
      )
```

Alternatively this can also be achieved with the `from Google.cloud import storage` module ([see the Google documentation for more information](https://cloud.Google.com/storage/docs/uploading-objects#storage-upload-object-code-sample)). To use this method comment out the code on [L180-L199](https://github.com/rickstaa/Pytorch_RL_on_google_AI_example/blob/8af3960064e1b67cfcc3efbdcbd020b3bb4c6153/dqn_basic.py#L180-L199) and [L271-L289](https://github.com/rickstaa/Pytorch_RL_on_google_AI_example/blob/8af3960064e1b67cfcc3efbdcbd020b3bb4c6153/dqn_basic.py#L271-L289) of the [dqn_basic.py](https://github.com/rickstaa/Pytorch_RL_on_google_AI_example/blob/master/dqn_basic.py) file.

### Docker file

The Docker file is used to create the RL training container we want to push to the Google AI platform. Most of the steps in the Docker File are used to setup the required dependencies and transfer the required script files. The most important component is the `ENTRYPOINT` at the bottom of the script:

```bash
ENTRYPOINT ["python3", "dqn_basic.py"]
```

This entry point makes sure our training script is executed when we deploy the docker image to the Google AI platform. After the docker image is deployed, the Google AI platform will allocate the required resources (CPU, GPU and memory) for running our training. These resources are detached again when the training script finishes. As a result, you only pay for the resources you used.

## Training in the cloud steps (CPU)

### Create a Google cloud bucket

To obtain the lowest training cost, you must place both the container registry, storage bucket and Google AI server in the same region. You therefore first have to choose a region in which you want to perform the training. An overview of the possible google computing regions and zones can be found [here](https://cloud.google.com/compute/docs/regions-zones/). An overview of the google container registry regions can be found [here](https://cloud.google.com/container-registry/docs/pushing-and-pulling). After you found your region you have to export them as environment variables:

```bash
export REGION=europe-west1
export CONTAINER_REGION=eu.gcr.io
```

Following you can create a Google cloud bucket:

```bash
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export BUCKET_NAME=${PROJECT_ID}-${REGION}-pytorch_dql_pong
gsutil mb -l ${REGION} gs://${BUCKET_NAME}
```

Please note that the cost you pay for storing/retrieving data in/from your bucket depends on the region you choose. An overview of the google cloud storage pricing can be found [here](https://cloud.google.com/storage/pricing).

### Build and test the container locally

Before pushing the containerized RL training job to the AI platform, we first want to test whether the container executes without errors. To do first export the following bash environmental variables:

```bash
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=pytorch_dql_pong_gpu_container
export IMAGE_TAG=dql_pytorch_gpu
export IMAGE_URI=eu.gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```

You are free to modify these variables in any way you like. After you set these variables, you can now build the docker image:

```bash
docker build -f Dockerfile -t $IMAGE_URI ./
```

If the container has successfully built, we can test it:

```bash
docker run $IMAGE_URI
```

### Push the container to the container Registry

If the container with your RL algorithm, is executing successfully, you can push it to the Google container repository:

```bash
docker push $IMAGE_URI
```

### Push the RL training job

Finally, we are ready to submit the training job to AI Platform Training using the gcloud tool:

```bash
export JOB_NAME=pytorch_dql_pong_job_$(date +%Y%m%d_%H%M%S)
export OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
 --region $REGION \
 --scale-tier BASIC \
 --master-image-uri $IMAGE_URI \
 -- \
 --no-cuda \
 --model-dir $OUTPUT_PATH \
```

The most important arguments are the following:

- `--region`: The region from which you want to use the computing resources.
- `--scale-tier`: This is the type of computing resource you use for performing the training job (see [ai-platform pricing page](https://cloud.Google.com/ai-platform/training/pricing) for more information).
- `--master-image-uri`: The URI to your Docker image.
- `--no-cuda`: Forces the CPU to be used even if GPU is available.
- `--model-dir`: The URI to the google cloud bucket.

### Check the results

After the training job has been deployed you can check visualize the results directly from the Google cloud bucket using the following tensorboard command:

## Training in the cloud steps (GPU)

To use GPU during training, you have to change the job submit command. In the job submit command change the `--scale-tier` option from `BASIC` to `BASIC_GPU` and add the `--cuda` command:

```bash
export JOB_NAME=pytorch_dql_pong_job_$(date +%Y%m%d_%H%M%S)
export OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
 --region $REGION \
 --scale-tier BASIC_GPU \
 --master-image-uri $IMAGE_URI \
 -- \
 --model-dir $OUTPUT_PATH
```

⚠️💰 Please keep in mind that changing the scale-tier from `BASIC` to `BASIC_GPU` increases the training cost! For an overview of the cost of training in the cloud see [the Google ai documentation](https://cloud.Google.com/ai-platform/training/pricing).

## Hyperparameter tuning

Additionally, as explained in the [google documentation](https://cloud.google.com/ai-platform/training/docs/using-containers), you can also perform hyperparameter tuning in the cloud. In this example, I try to tune the `batch_size` hyperparameter. You can push a hyperparameter training job to the Google Ai cloud by supplying the job submit command with the hyperparameter `config.yaml` file:

```bash
export JOB_NAME=pytorch_dql_pong_job_$(date +%Y%m%d_%H%M%S)
export OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --scale-tier BASIC \
  --master-image-uri $IMAGE_URI \
  --config config.yaml \
  -- \
  --no-cuda \
  --model-dir $OUTPUT_PATH
```

### Clean up

When your finished with this example you can delete the Google cloud bucket using the following command:

```bash
gsutil rm -r gs://${PROJECT}-singularity
```
