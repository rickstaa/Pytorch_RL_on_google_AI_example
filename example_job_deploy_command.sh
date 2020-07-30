gcloud ai-platform jobs submit training $JOB_NAME \
	--region $REGION \
	--scale-tier BASIC_GPU \
	--master-image-uri $IMAGE_URI \
	-- \
	--cuda \
	--model-dir $OUTPUT_PATH \